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
    "context7": {
        "command": "npx",
        "args": ["-y", "@upstash/context7-mcp@latest"],
        "transport": "stdio",
    },
    
    # 예시 2: Sequential Thinking MCP 서버 (단계별 사고)
    # 복잡한 문제를 단계별로 분석하는 사고 도구를 제공합니다.
    "sequential_thinking": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "transport": "stdio",
    },
    
    # 예시 3: 커스텀 로컬 MCP 서버 (Python 기반)
    # 직접 만든 MCP 서버를 연결할 때 사용합니다.
    # "custom_server": {
    #     "command": "python",
    #     "args": ["/absolute/path/to/your/mcp_server.py"],
    #     "transport": "stdio",
    # },
    
    # 예시 4: 원격 MCP 서버 (SSE 방식)
    # 이미 실행 중인 HTTP 기반 MCP 서버에 연결할 때 사용합니다.
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
        tuple: (client, agent) - MCP 클라이언트와 에이전트
        
    💡 MultiServerMCPClient는 async context manager로 사용해야 합니다.
       with 블록 안에서만 MCP 서버 연결이 유지됩니다.
    """
    settings = get_settings()
    
    # LLM 모델 초기화 (도구 바인딩은 create_react_agent가 처리)
    model = ChatOpenAI(
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )
    
    # MCP 클라이언트 생성 및 연결
    # MultiServerMCPClient는 여러 MCP 서버를 동시에 관리합니다.
    client = MultiServerMCPClient(server_configs)
    
    # 컨텍스트 매니저 진입 (서버 연결 시작)
    await client.__aenter__()
    
    # MCP 서버에서 제공하는 모든 도구 가져오기
    # get_tools()는 연결된 모든 서버의 도구를 LangChain Tool 형태로 반환합니다.
    tools = client.get_tools()
    
    print(f"📦 [MCP] 연결된 서버: {list(server_configs.keys())}")
    print(f"🔧 [MCP] 사용 가능한 도구: {[t.name for t in tools]}")
    
    # 시스템 프롬프트: 에이전트의 역할 정의
    system_prompt = """당신은 MCP 도구를 활용하여 사용자를 돕는 유능한 AI 어시스턴트입니다.
    
사용 가능한 도구를 적극적으로 활용하여 정확한 정보를 제공하세요.
모든 답변은 한국어로 친절하게 해주세요."""
    
    # create_react_agent로 에이전트 생성
    # MCP에서 가져온 도구를 그대로 전달합니다.
    agent = create_react_agent(
        model,
        tools=tools,
        prompt=system_prompt,
    )
    
    return client, agent


# =============================================================================
# ▶️ 3. 에이전트 실행 함수
# =============================================================================

async def run_mcp_agent(query: str, server_configs: dict = None):
    """
    MCP 에이전트를 실행하여 사용자 질문에 답변합니다.
    
    Args:
        query: 사용자 질문
        server_configs: MCP 서버 설정 (기본값: MCP_SERVER_CONFIGS)
    """
    # 서버 설정이 없으면 기본 설정 사용
    if server_configs is None:
        server_configs = MCP_SERVER_CONFIGS
    
    print(f"\n{'='*60}")
    print(f"🙋 사용자: {query}")
    print('='*60)
    
    client = None
    try:
        # MCP 에이전트 생성 (서버 연결 포함)
        client, agent = await create_mcp_agent(server_configs)
        
        # 에이전트 실행
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]}
        )
        
        # 결과에서 마지막 메시지(AI 응답) 추출
        if result.get("messages"):
            final_msg = result["messages"][-1]
            print(f"\n🤖 Agent: {final_msg.content}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("팁: MCP 서버가 올바르게 설정되었는지 확인하세요.")
        raise
        
    finally:
        # MCP 클라이언트 연결 종료
        if client:
            await client.__aexit__(None, None, None)


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
