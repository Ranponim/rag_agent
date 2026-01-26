# -*- coding: utf-8 -*-
"""
============================================================================
📚 01d. MCP Agent - Model Context Protocol 서버 연동 예제
============================================================================

이 예제는 외부 MCP(Model Context Protocol) 서버에 연결하여 
해당 서버가 제공하는 도구를 LangGraph 에이전트에서 활용하는 방법을 보여줍니다.

🎯 학습 목표:
    1. MCP 서버 등록 및 연결 방법 이해
    2. MultiServerMCPClient를 통한 다중 MCP 서버 관리
    3. MCP 도구를 LangGraph 에이전트에 바인딩하는 패턴

💡 핵심 개념:
    - MCP (Model Context Protocol): AI 모델이 외부 도구/리소스에 접근하는 표준 프로토콜
    - MultiServerMCPClient: 여러 MCP 서버를 동시에 관리하는 클라이언트
    - Transport: MCP 서버와 통신하는 방식 (stdio, sse, streamable-http)

📦 필수 패키지:
    pip install langchain-mcp-adapters langgraph

실행 방법:
    python examples/01d_mcp_agent.py
    
⚠️ 주의사항:
    - MCP 서버가 미리 실행 중이거나, command로 시작 가능해야 합니다.
    - 아래 예제의 서버 설정은 사용 환경에 맞게 수정하세요.
"""

# =============================================================================
# 📦 필수 라이브러리 임포트
# =============================================================================

import sys
import asyncio
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangGraph 프리빌트 컴포넌트
from langgraph.prebuilt import create_react_agent

# LangChain 컴포넌트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# MCP 어댑터 (MCP 서버 연결용)
# pip install langchain-mcp-adapters
from langchain_mcp_adapters.client import MultiServerMCPClient

# 프로젝트 설정 로드
from config.settings import get_settings

# MCP 클라이언트 관리 유틸리티 (오류 처리 및 재시도 로직 포함)
from utils.mcp_client import MCPClientManager


# =============================================================================
# ⚙️ 1. MCP 서버 설정 정의
# =============================================================================
# 
# MCP 서버는 크게 두 가지 방식으로 연결할 수 있습니다:
# 
# 1. stdio (Standard I/O): 로컬 프로세스로 서버를 실행
#    - command: 실행할 명령어 (예: "python", "npx", "node")
#    - args: 명령어 인자 (예: ["/path/to/server.py"])
#    - transport: "stdio"
#
# 2. SSE (Server-Sent Events): 원격 HTTP 서버에 연결
#    - url: 서버 URL (예: "http://localhost:8000/sse")
#    - transport: "sse"
#
# 💡 본 예제에서는 자주 사용되는 MCP 서버들의 설정 예시를 제공합니다.
# =============================================================================

# MCP 서버 설정 딕셔너리
# 키(key)는 서버 식별자로, 원하는 이름을 지정할 수 있습니다.
MCP_SERVER_CONFIGS = {
    # 예시 1: Context7 MCP 서버 (라이브러리 문서 검색)
    # npx를 통해 자동으로 패키지를 다운로드하고 실행합니다.
    # Transport: stdio (로컬 프로세스로 실행)
    "context7": {
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp@latest"],
        "transport": "stdio",
    },
    
    # 예시 2: Sequential Thinking MCP 서버 (단계별 사고)
    # 복잡한 문제를 단계별로 분석하는 사고 도구를 제공합니다.
    # Transport: stdio (로컬 프로세스로 실행)
    "sequential_thinking": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "transport": "stdio",
    },
    
    # 예시 3: Analysis LLM MCP 서버 (3GPP 분석 도구)
    # Docker 환경에서 실행 중인 MCP 서버에 연결합니다.
    # Transport: streamable_http (HTTP 스트리밍 방식)
    # ⚠️ 주의: 서버가 http://165.213.69.30:8001/mcp 에서 실행 중이어야 합니다.
    "analysis_llm": {
        "transport": "streamable_http", 
        "url": "http://165.213.69.30:8001/mcp",  # /mcp 엔드포인트로 복구
        # 인증이 필요한 경우 아래 주석을 해제하고 토큰을 설정하세요.
        # "headers": {
        #     "Authorization": "Bearer YOUR_API_TOKEN",
        #     "X-Custom-Header": "custom-value"
        # },
    },
    
    # 예시 4: 커스텀 로컬 MCP 서버 (Python 기반)
    # 직접 만든 MCP 서버를 연결할 때 사용합니다.
    # Transport: stdio (로컬 프로세스로 실행)
    # "custom_server": {
    #     "command": "python",
    #     "args": ["/absolute/path/to/your/mcp_server.py"],
    #     "transport": "stdio",
    # },
    
    # 예시 5: 원격 MCP 서버 (SSE 방식)
    # 이미 실행 중인 서버에 Server-Sent Events로 연결할 때 사용합니다.
    # Transport: sse (서버 푸시 기반 통신)
    # "remote_server": {
    #     "url": "http://localhost:8000/sse",
    #     "transport": "sse",
    # },
}


# =============================================================================
# 🤖 2. MCP 에이전트 생성 함수
# =============================================================================

async def create_mcp_agent(server_configs: dict):
    """
    MCP 서버에 연결하고, 해당 도구를 사용하는 에이전트를 생성합니다.
    
    Args:
        server_configs: MCP 서버 설정 딕셔너리
        
    Returns:
        tuple: (manager, agent) - MCP 클라이언트 매니저와 에이전트
        
    💡 MCPClientManager를 사용하여 연결 관리, 오류 처리, 재시도 등을 자동화합니다.
       반환된 manager는 반드시 disconnect()를 호출하여 정리해야 합니다.
    """
    # 설정 로드
    settings = get_settings()
    
    # LLM 모델 초기화
    # create_react_agent가 내부적으로 도구를 바인딩하므로
    # 여기서는 모델만 생성합니다.
    model = ChatOpenAI(
        base_url=settings.openai_api_base,  # OpenAI 호환 API 엔드포인트
        api_key=settings.openai_api_key,    # API 인증 키
        model=settings.openai_model,        # 사용할 모델 이름
    )
    
    print(f"\n{'='*70}")
    print(f"🤖 [Agent] LLM 모델 초기화: {settings.openai_model}")
    print(f"🌐 [Agent] API Base: {settings.openai_api_base}")
    print(f"{'='*70}\n")
    
    # MCP 클라이언트 매니저 생성 및 연결
    # MCPClientManager는 연결 관리, 오류 처리, 재시도를 자동으로 처리합니다.
    manager = MCPClientManager(
        server_configs=server_configs,  # MCP 서버 설정
        max_retries=3,                  # 연결 실패 시 최대 3회 재시도
        retry_delay=1.0                 # 재시도 간 1초 대기 (exponential backoff 적용)
    )
    
    # 서버 연결 시도 (내부적으로 재시도 로직 포함)
    # 연결 실패 시 예외가 발생하므로 호출하는 쪽에서 try-except로 처리해야 합니다.
    await manager.connect()
    
    # MCP 서버에서 제공하는 모든 도구 가져오기
    # get_tools()는 연결된 모든 서버의 도구를 LangChain Tool 형태로 반환합니다.
    tools = await manager.get_tools()
    
    # 연결 정보 출력
    print(f"\n{'='*70}")
    print(f"📦 [MCP] 연결된 서버: {list(server_configs.keys())}")
    print(f"🔧 [MCP] 총 {len(tools)}개의 도구 사용 가능")
    print(f"{'='*70}\n")
    
    # 시스템 프롬프트: 에이전트의 역할 및 지침 정의
    system_prompt = """당신은 MCP(Model Context Protocol) 도구를 활용하여 사용자를 돕는 유능한 AI 어시스턴트입니다.
    
주요 역할:
- 사용 가능한 도구를 적극적으로 활용하여 정확한 정보를 제공합니다.
- 복잡한 질문은 단계별로 분해하여 처리합니다.
- 도구 실행 결과를 바탕으로 신뢰할 수 있는 답변을 제공합니다.

답변 원칙:
- 모든 답변은 한국어로 친절하게 작성합니다.
- 불확실한 정보는 추측하지 말고 도구를 사용하여 확인합니다.
- 도구 실행 중 오류가 발생하면 사용자에게 명확히 설명합니다."""
    
    # create_react_agent로 ReAct 패턴 에이전트 생성
    # ReAct: Reasoning과 Acting을 반복하여 문제를 해결하는 패턴
    # - Reasoning: LLM이 다음 행동을 결정
    # - Acting: 도구를 실행하거나 최종 답변 생성
    agent = create_react_agent(
        model,              # LLM 모델 (ChatOpenAI 인스턴스)
        tools=tools,        # MCP 서버에서 가져온 도구 리스트
        prompt=system_prompt,  # 시스템 프롬프트 (에이전트의 역할 정의)
    )
    
    print(f"✅ [Agent] ReAct 에이전트 생성 완료\n")
    
    # MCP 클라이언트 매니저와 에이전트를 함께 반환
    # 매니저는 나중에 연결을 종료하는 데 필요합니다.
    return manager, agent


# =============================================================================
# ▶️ 3. 에이전트 실행 함수
# =============================================================================

async def run_mcp_agent(query: str, server_configs: dict = None):
    """
    MCP 에이전트를 실행하여 사용자 질문에 답변합니다.
    
    이 함수는 다음 단계를 수행합니다:
    1. MCP 서버에 연결
    2. 도구를 가져와 에이전트 생성
    3. 사용자 질문을 에이전트에 전달
    4. 결과를 출력
    5. 연결 종료 (finally 블록에서 안전하게 처리)
    
    Args:
        query (str): 사용자 질문
        server_configs (dict, optional): MCP 서버 설정. None이면 기본 설정 사용
    
    Raises:
        Exception: MCP 연결 실패, 도구 가져오기 실패, 에이전트 실행 실패 시
    
    Example:
        >>> await run_mcp_agent("LangGraph의 주요 기능을 알려줘")
    """
    # 서버 설정이 없으면 기본 설정 사용
    if server_configs is None:
        server_configs = MCP_SERVER_CONFIGS
    
    # 사용자 질문 출력 (시각적 구분을 위해 구분선 사용)
    print(f"\n{'='*70}")
    print(f"🙋 사용자 질문: {query}")
    print('='*70)
    
    # MCP 클라이언트 매니저 (finally에서 연결 종료를 위해 변수 선언)
    manager = None
    
    try:
        # ========================================
        # 1단계: MCP 에이전트 생성 (서버 연결 포함)
        # ========================================
        print("\n[1/3] MCP 서버 연결 및 에이전트 생성 중...")
        manager, agent = await create_mcp_agent(server_configs)
        
        # ========================================
        # 2단계: 에이전트 실행 (스트리밍 방식)
        # ========================================
        print("[2/3] 에이전트 실행 중...\n")
        
        # astream()을 사용하여 각 단계를 실시간으로 확인
        # 이를 통해 어떤 도구가 선택되었는지, 어떤 파라미터로 호출되었는지 추적 가능
        final_response = None
        step_count = 0
        
        # HumanMessage로 사용자 입력을 감싸서 전달
        async for chunk in agent.astream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="values"  # 전체 상태를 반환 (메시지 리스트 포함)
        ):
            # 각 chunk는 현재 상태의 스냅샷
            # messages 키에 현재까지의 모든 메시지가 담겨 있음
            if "messages" in chunk:
                messages = chunk["messages"]
                
                # 마지막 메시지 확인
                if messages:
                    last_msg = messages[-1]
                    
                    # AI 메시지인지 확인 (도구 호출 또는 최종 응답)
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        # 도구 호출이 있는 경우
                        step_count += 1
                        print(f"\n🔧 [Step {step_count}] 도구 호출 감지:")
                        
                        for tool_call in last_msg.tool_calls:
                            # 도구 이름 출력
                            tool_name = tool_call.get('name', 'Unknown')
                            print(f"  📌 도구: {tool_name}")
                            
                            # 도구 파라미터 출력
                            tool_args = tool_call.get('args', {})
                            if tool_args:
                                print(f"  📝 파라미터:")
                                for key, value in tool_args.items():
                                    # 값이 너무 길면 잘라서 표시
                                    value_str = str(value)
                                    if len(value_str) > 100:
                                        value_str = value_str[:100] + "..."
                                    print(f"     - {key}: {value_str}")
                            
                            print()  # 빈 줄 추가
                    
                    # ToolMessage인지 확인 (도구 실행 결과)
                    elif hasattr(last_msg, '__class__') and last_msg.__class__.__name__ == 'ToolMessage':
                        print(f"✅ [Step {step_count}] 도구 실행 완료")
                        
                        # 도구 실행 결과 출력 (너무 길면 생략)
                        content = last_msg.content
                        if len(content) > 200:
                            print(f"  📊 결과: {content[:200]}...\n")
                        else:
                            print(f"  📊 결과: {content}\n")
                
                # 최종 응답 저장
                final_response = chunk
        
        # ========================================
        # 3단계: 결과 출력
        # ========================================
        print(f"\n{'='*70}")
        print(f"[3/3] 실행 완료 (총 {step_count}개 도구 호출)")
        print(f"{'='*70}\n")
        
        # 최종 응답 출력
        if final_response and final_response.get("messages"):
            final_msg = final_response["messages"][-1]
            
            # 최종 메시지가 AI 응답인 경우
            if hasattr(final_msg, 'content') and final_msg.content:
                print(f"🤖 AI 최종 응답:\n{final_msg.content}\n")
            else:
                print("⚠️ 경고: 최종 응답이 비어있습니다.")
        else:
            # 메시지가 없는 경우 (예상치 못한 상황)
            print("⚠️ 경고: 에이전트 응답이 비어있습니다.")
        
    except ValueError as e:
        # 설정 검증 오류 (서버 설정이 잘못된 경우)
        print(f"\n❌ 설정 오류: {e}")
        print("💡 해결 방법:")
        print("   - MCP_SERVER_CONFIGS의 각 서버 설정을 확인하세요.")
        print("   - transport, url, command 등 필수 필드가 올바른지 확인하세요.")
        raise
        
    except ConnectionError as e:
        # 네트워크 연결 오류 (서버에 접근할 수 없는 경우)
        print(f"\n❌ 연결 오류: {e}")
        print("💡 해결 방법:")
        print("   - MCP 서버가 실행 중인지 확인하세요.")
        print("   - 네트워크 연결과 방화벽 설정을 확인하세요.")
        print("   - URL이 올바른지 확인하세요.")
        raise
        
    except Exception as e:
        # 기타 모든 예외 처리
        print(f"\n❌ 예기치 않은 오류 발생: {type(e).__name__}")
        print(f"오류 메시지: {e}")
        print("\n💡 일반적인 해결 방법:")
        print("   1. MCP 서버가 올바르게 설정되었는지 확인")
        print("   2. 필요한 패키지가 설치되었는지 확인 (langchain-mcp-adapters)")
        print("   3. Python 버전 호환성 확인")
        print(f"\n📋 상세 오류 정보:")
        import traceback
        traceback.print_exc()  # 전체 스택 트레이스 출력
        raise
        
    finally:
        # ========================================
        # 리소스 정리: MCP 클라이언트 연결 종료
        # ========================================
        # finally 블록은 예외 발생 여부와 관계없이 항상 실행됩니다.
        # 이를 통해 리소스 누수를 방지합니다.
        if manager:
            print("\n[정리] MCP 서버 연결 종료 중...")
            await manager.disconnect()
            print("✅ 연결 안전하게 종료됨\n")


# =============================================================================
# 🔄 4. 간단 사용 예시 (단일 서버 연결)
# =============================================================================

async def simple_mcp_example():
    """
    단일 MCP 서버(Context7)만 연결하는 간단한 예제입니다.
    
    💡 async with 문법을 사용하면 자동으로 연결 종료가 처리됩니다.
    """
    settings = get_settings()
    
    # LLM 모델 초기화
    model = ChatOpenAI(
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )
    
    # Context7 MCP 서버만 연결하는 간단한 예제
    async with MultiServerMCPClient(
        {
            "context7": {
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp@latest"],
                "transport": "stdio",
            }
        }
    ) as client:
        # MCP 도구 가져오기
        tools = client.get_tools()
        print(f"🔧 사용 가능한 도구: {[t.name for t in tools]}")
        
        # 에이전트 생성
        agent = create_react_agent(
            model,
            tools=tools,
            prompt="당신은 라이브러리 문서를 검색하는 전문가입니다."
        )
        
        # 질문 실행
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="LangGraph의 주요 기능을 알려줘")]}
        )
        
        if result.get("messages"):
            print(f"\n🤖 응답: {result['messages'][-1].content}")


# =============================================================================
# 🚀 5. 메인 실행부
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 LangGraph MCP Agent Example")
    print("="*60)
    
    # 테스트 질문
    # Context7 MCP를 사용하여 라이브러리 문서를 검색하는 예제
    test_query = "LangGraph의 create_react_agent 함수 사용법을 알려줘"
    
    # 비동기 실행
    # asyncio.run()으로 async 함수를 실행합니다.
    try:
        asyncio.run(run_mcp_agent(test_query))
    except KeyboardInterrupt:
        print("\n👋 종료합니다.")
