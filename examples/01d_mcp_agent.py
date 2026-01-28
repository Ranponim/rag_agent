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
import os
import asyncio
from pathlib import Path

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangGraph 프리빌트 컴포넌트
from langgraph.prebuilt import create_react_agent

# LangChain 컴포넌트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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
    # "context7": {
    #     "command": "npx",
    #     "args": ["-y", "@upstash/context7-mcp@latest"],
    #     "transport": "stdio",
    # },
    
    # 예시 2: Sequential Thinking MCP 서버 (단계별 사고)
    # 복잡한 문제를 단계별로 분석하는 사고 도구를 제공합니다.
    # Transport: stdio (로컬 프로세스로 실행)
    "sequential_thinking": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "transport": "stdio",
    },
    
    # 예시 3: Analysis LLM MCP 서버 (3GPP 분석 도구)
    # 원격 IP(165...)는 Python 환경에서 접근 불가하므로 localhost를 타겟으로 합니다.
    # Transport: streamable_http (HTTP 스트리밍 방식)
    # "analysis_llm": {
    #     "transport": "streamable_http", 
    #     "url": "http://localhost:8001/mcp",  # localhost 주소로 변경
    #     # PowerShell 성공 시 사용된 헤더를 MCPClientManager가 자동 주입합니다.
    #     "headers": {
    #         "Accept": "application/json, text/event-stream"
    #     },
    # },
    
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
    
    # [NEW] 로컬 PC 디렉토리 탐색 MCP (FastMCP)
    "directory_explorer": {
        "command": "python",
        "args": [str(Path(__file__).parent.parent / "mcp" / "simple_dir_mcp.py")],
        "transport": "stdio",
    },
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
    # LLM 모델 초기화
    # 환경변수에서 값을 가져와 변수에 할당 (print문에서 사용하기 위함)
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")
    
    model = ChatOpenAI(
        base_url=api_base,
        api_key=api_key,
        model=model_name
    )
    
    print(f"\n{'='*70}")
    print(f"🤖 [Agent] LLM 모델 초기화: {model_name}")
    print(f"🌐 [Agent] API Base: {api_base}")
    print(f"{'='*70}\n")
    
    # MCP 클라이언트 매니저 생성 및 연결
    # MCPClientManager는 PowerShell 성공 사례의 헤더(Connection: keep-alive 등)를 
    # 자동으로 주입하여 RemoteProtocolError를 방지합니다.
    manager = MCPClientManager(
        server_configs=server_configs,
        max_retries=3,
        retry_delay=2.0  # 서버 응답 대기 시간을 고려하여 지연 시간 상향
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
# 🔄 4. 대화형 실행 함수 (CLI Chat)
# =============================================================================

async def run_interactive_mcp_agent(server_configs: dict = None):
    """
    사용자와 대화하며 MCP 에이전트를 실행하는 대화형 루프입니다.
    연결을 유지한 상태로 연속적인 대화가 가능합니다.
    """
    if server_configs is None:
        server_configs = MCP_SERVER_CONFIGS

    print(f"\n{'='*70}")
    print("💬 MCP Interactive Chat Mode")
    print(f"{'='*70}")
    print("MCP 서버에 연결하고 에이전트를 초기화합니다...\n")

    manager = None
    
    try:
        # 1. 초기화 (한 번만 수행)
        manager, app = await create_mcp_agent(server_configs)
        
        # 대화 기록 유지
        chat_history = []
        
        print("\n✅ 준비 완료! 대화를 시작하세요. (종료하려면 'q' 또는 'quit' 입력)")
        print(f"{'-'*70}\n")
        
        while True:
            try:
                # 사용자 입력
                query = input("\n🙋 User: ").strip()
                if not query:
                    continue
                    
                if query.lower() in ['q', 'quit', 'exit']:
                    print("\n👋 대화를 종료합니다.")
                    break
                
                # 메시지 구성 (기존 히스토리 + 새 질문)
                current_messages = chat_history + [HumanMessage(content=query)]
                
                print(f"\n🤖 Agent 생각 중...", end="", flush=True)
                
                # 스트리밍 실행
                step_count = 0
                final_response_chunk = None
                
                # astream을 사용하여 실행 과정 시각화
                async for chunk in app.astream(
                    {"messages": current_messages},
                    stream_mode="values"
                ):
                    if "messages" in chunk:
                        messages = chunk["messages"]
                        if messages:
                            last_msg = messages[-1]
                            
                            # 도구 호출 로깅
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                step_count += 1
                                print(f"\n\n🔧 [Step {step_count}] 도구 호출:")
                                for tool_call in last_msg.tool_calls:
                                    print(f"  📌 {tool_call.get('name')}: {tool_call.get('args')}")
                                print("  ⏳ 실행 중...", end="", flush=True)
                            
                            # 도구 결과 로깅
                            elif hasattr(last_msg, 'content') and last_msg.content and len(messages) > len(current_messages):
                                # AI의 중간 응답이나 최종 응답이 아닐 때 (즉, ToolMessage 바로 다음이 아닌 경우 등)
                                pass

                        final_response_chunk = chunk

                # 최종 응답 처리
                if final_response_chunk and "messages" in final_response_chunk:
                    final_messages = final_response_chunk["messages"]
                    last_msg = final_messages[-1]
                    
                    if hasattr(last_msg, 'content') and last_msg.content:
                        print(f"\n\n🤖 Agent:\n{last_msg.content}\n")
                    
                    # 대화 기록 업데이트 (전체 히스토리 덮어쓰기)
                    chat_history = final_messages
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ 인터럽트 감지. 대화를 종료합니다.")
                break
            except Exception as e:
                print(f"\n\n❌ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"\n❌ 초기화 오류: {e}")
    finally:
        if manager:
            print("\n🔌 연결 종료 중...")
            await manager.disconnect()
            print("✅ 연결 종료 완료")


# =============================================================================
# 🚀 5. 메인 실행부
# =============================================================================

if __name__ == "__main__":
    # 비동기 실행
    try:
        # CLI 채팅 모드 실행
        asyncio.run(run_interactive_mcp_agent())
    except KeyboardInterrupt:
        print("\n👋 종료합니다.")
