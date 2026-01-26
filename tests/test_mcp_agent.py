# -*- coding: utf-8 -*-
"""
============================================================================
🧪 MCP Agent 테스트 - MCP 서버 연결 및 도구 사용 검증
============================================================================

이 테스트 스크립트는 MCP 서버 연결과 도구 사용이 정상적으로 작동하는지 검증합니다.

🎯 테스트 항목:
    1. MCP 서버 연결 테스트
    2. 도구 목록 가져오기 테스트
    3. 에이전트 통합 테스트
    4. 오류 처리 테스트

⚠️ 주의사항:
    - 실제 MCP 서버가 실행 중이어야 테스트가 성공합니다.
    - 네트워크 연결이 가능해야 합니다.
    - HTTP 기반 MCP 서버는 http://165.213.69.30:8001/mcp에서 실행 중이어야 합니다.

실행 방법:
    python tests/test_mcp_agent.py
"""

# =============================================================================
# 📦 필수 라이브러리 임포트
# =============================================================================

import sys
import asyncio
from pathlib import Path

# 프로젝트 루트를 경로에 추가
# tests/ 디렉토리에서 실행하므로 상위 디렉토리를 추가해야 합니다.
sys.path.insert(0, str(Path(__file__).parent.parent))

# MCP 클라이언트 관리 유틸리티
from utils.mcp_client import MCPClientManager

# 설정 로드
from config.settings import get_settings

# LangChain 컴포넌트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# LangGraph 에이전트
from langgraph.prebuilt import create_react_agent


# =============================================================================
# 🔧 테스트용 MCP 서버 설정
# =============================================================================

# analysis_llm MCP 서버만 테스트 (HTTP 기반)
# 다른 서버들은 주석 처리하여 테스트 시간 단축
TEST_SERVER_CONFIGS = {
    # HTTP 기반 analysis_llm MCP 서버
    # Docker 환경에서 실행 중인 서버에 연결
    "analysis_llm": {
        "transport": "streamable_http",  # HTTP 스트리밍 방식
        "url": "http://165.213.69.30:8001/mcp",  # MCP 서버 엔드포인트
    },
    
    # 필요시 다른 서버도 추가 가능
    # "context7": {
    #     "command": "npx",
    #     "args": ["-y", "@upstash/context7-mcp@latest"],
    #     "transport": "stdio",
    # },
}


# =============================================================================
# 🧪 테스트 1: MCP 서버 연결 테스트
# =============================================================================

async def test_mcp_connection():
    """
    MCP 서버 연결이 정상적으로 수행되는지 테스트합니다.
    
    검증 항목:
    - 서버 연결 성공
    - 연결 상태 확인
    - 서버 정보 조회
    
    Returns:
        bool: 테스트 성공 여부
    """
    print("\n" + "="*70)
    print("🧪 [TEST 1] MCP 서버 연결 테스트")
    print("="*70 + "\n")
    
    manager = None
    try:
        # MCP 클라이언트 매니저 생성
        print("1️⃣ MCPClientManager 생성 중...")
        manager = MCPClientManager(
            server_configs=TEST_SERVER_CONFIGS,
            max_retries=3,
            retry_delay=1.0
        )
        print("   ✅ MCPClientManager 생성 완료\n")
        
        # 서버 연결
        print("2️⃣ MCP 서버 연결 시도 중...")
        await manager.connect()
        print("   ✅ 서버 연결 성공\n")
        
        # 연결 상태 확인
        print("3️⃣ 연결 상태 확인 중...")
        if manager.is_connected:
            print("   ✅ 연결 상태: 정상\n")
        else:
            print("   ❌ 연결 상태: 비정상\n")
            return False
        
        # 서버 정보 조회
        print("4️⃣ 서버 정보 조회 중...")
        server_info = manager.get_server_info()
        print(f"   📊 등록된 서버 수: {len(server_info)}")
        for name, config in server_info.items():
            print(f"   📡 {name}: {config.get('transport')} - {config.get('url', 'N/A')}")
        print()
        
        print("✅ [TEST 1] 통과: MCP 서버 연결 성공\n")
        return True
        
    except Exception as e:
        print(f"❌ [TEST 1] 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 연결 종료
        if manager:
            print("🔌 연결 종료 중...")
            await manager.disconnect()
            print("✅ 연결 종료 완료\n")


# =============================================================================
# 🧪 테스트 2: 도구 목록 가져오기 테스트
# =============================================================================

async def test_get_tools():
    """
    MCP 서버에서 도구 목록을 정상적으로 가져오는지 테스트합니다.
    
    검증 항목:
    - 도구 목록 조회 성공
    - 도구 개수 확인
    - 각 도구의 속성 확인 (이름, 설명)
    
    Returns:
        bool: 테스트 성공 여부
    """
    print("\n" + "="*70)
    print("🧪 [TEST 2] 도구 목록 가져오기 테스트")
    print("="*70 + "\n")
    
    manager = None
    try:
        # MCP 클라이언트 매니저 생성 및 연결
        print("1️⃣ MCP 서버 연결 중...")
        manager = MCPClientManager(
            server_configs=TEST_SERVER_CONFIGS,
            max_retries=3,
            retry_delay=1.0
        )
        await manager.connect()
        print("   ✅ 연결 완료\n")
        
        # 도구 목록 가져오기
        print("2️⃣ 도구 목록 조회 중...")
        tools = await manager.get_tools()
        print(f"   ✅ {len(tools)}개의 도구 발견\n")
        
        # 도구가 있는지 확인
        if len(tools) == 0:
            print("   ❌ 도구가 하나도 없습니다.\n")
            return False
        
        # 각 도구의 정보 출력
        print("3️⃣ 도구 세부 정보:")
        for i, tool in enumerate(tools, 1):
            print(f"   {i}. 🔨 {tool.name}")
            print(f"      📝 설명: {tool.description[:100]}...")  # 설명 일부만 출력
            print(f"      📋 타입: {type(tool).__name__}")
            print()
        
        print("✅ [TEST 2] 통과: 도구 목록 가져오기 성공\n")
        return True
        
    except Exception as e:
        print(f"❌ [TEST 2] 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 연결 종료
        if manager:
            await manager.disconnect()


# =============================================================================
# 🧪 테스트 3: 에이전트 통합 테스트
# =============================================================================

async def test_agent_with_mcp():
    """
    MCP 도구를 사용하는 에이전트가 정상적으로 작동하는지 테스트합니다.
    
    검증 항목:
    - 에이전트 생성 성공
    - 간단한 질문에 대한 응답 생성
    - 도구 호출 여부 (선택사항)
    
    Returns:
        bool: 테스트 성공 여부
    """
    print("\n" + "="*70)
    print("🧪 [TEST 3] 에이전트 통합 테스트")
    print("="*70 + "\n")
    
    manager = None
    try:
        # 설정 로드
        settings = get_settings()
        
        # LLM 모델 초기화
        print("1️⃣ LLM 모델 초기화 중...")
        model = ChatOpenAI(
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        print(f"   ✅ 모델: {settings.openai_model}\n")
        
        # MCP 서버 연결 및 도구 가져오기
        print("2️⃣ MCP 서버 연결 중...")
        manager = MCPClientManager(
            server_configs=TEST_SERVER_CONFIGS,
            max_retries=3,
            retry_delay=1.0
        )
        await manager.connect()
        tools = await manager.get_tools()
        print(f"   ✅ {len(tools)}개 도구 로드 완료\n")
        
        # 에이전트 생성
        print("3️⃣ ReAct 에이전트 생성 중...")
        agent = create_react_agent(
            model,
            tools=tools,
            prompt="당신은 MCP 도구를 사용하여 질문에 답변하는 AI 어시스턴트입니다. 한국어로 답변하세요."
        )
        print("   ✅ 에이전트 생성 완료\n")
        
        # 간단한 질문 실행
        print("4️⃣ 에이전트 실행 테스트 중...")
        test_query = "안녕하세요. 간단한 인사말에 답변해주세요."
        print(f"   📝 질문: {test_query}\n")
        
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=test_query)]}
        )
        
        # 응답 확인
        if result.get("messages"):
            final_msg = result["messages"][-1]
            print(f"   🤖 응답: {final_msg.content[:200]}...\n")  # 응답 일부만 출력
            print("   ✅ 에이전트 응답 생성 성공\n")
        else:
            print("   ❌ 에이전트 응답이 없습니다.\n")
            return False
        
        print("✅ [TEST 3] 통과: 에이전트 통합 테스트 성공\n")
        return True
        
    except Exception as e:
        print(f"❌ [TEST 3] 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 연결 종료
        if manager:
            await manager.disconnect()


# =============================================================================
# 🧪 테스트 4: 오류 처리 테스트
# =============================================================================

async def test_error_handling():
    """
    잘못된 설정에 대한 오류 처리가 정상적으로 작동하는지 테스트합니다.
    
    검증 항목:
    - 잘못된 URL에 대한 연결 실패 처리
    - 누락된 필드에 대한 검증 오류 처리
    
    Returns:
        bool: 테스트 성공 여부
    """
    print("\n" + "="*70)
    print("🧪 [TEST 4] 오류 처리 테스트")
    print("="*70 + "\n")
    
    # 테스트 4-1: 잘못된 URL
    print("4-1️⃣ 잘못된 URL 테스트...")
    invalid_url_config = {
        "invalid_server": {
            "transport": "streamable_http",
            "url": "http://invalid-url-that-does-not-exist.com/mcp",
        }
    }
    
    manager = None
    try:
        manager = MCPClientManager(
            server_configs=invalid_url_config,
            max_retries=1,  # 빠른 테스트를 위해 재시도 1회만
            retry_delay=0.5
        )
        await manager.connect()
        print("   ❌ 예외가 발생하지 않았습니다. (예상: 연결 실패)\n")
        return False
        
    except Exception as e:
        print(f"   ✅ 예상대로 예외 발생: {type(e).__name__}\n")
        
    finally:
        if manager:
            await manager.disconnect()
    
    # 테스트 4-2: 누락된 필수 필드
    print("4-2️⃣ 누락된 필수 필드 테스트...")
    invalid_config = {
        "invalid_server": {
            # transport 필드 누락
            "url": "http://localhost:8000/mcp",
        }
    }
    
    try:
        manager = MCPClientManager(server_configs=invalid_config)
        print("   ❌ 예외가 발생하지 않았습니다. (예상: ValueError)\n")
        return False
        
    except ValueError as e:
        print(f"   ✅ 예상대로 검증 오류 발생: {e}\n")
        
    except Exception as e:
        print(f"   ⚠️ 다른 예외 발생: {type(e).__name__}: {e}\n")
    
    print("✅ [TEST 4] 통과: 오류 처리 테스트 성공\n")
    return True


# =============================================================================
# 🚀 메인 실행부
# =============================================================================

async def main():
    """
    모든 테스트를 순차적으로 실행합니다.
    
    각 테스트의 성공/실패를 추적하고 최종 결과를 출력합니다.
    """
    print("\n" + "🎯"*35)
    print("🧪 MCP Agent 테스트 시작")
    print("🎯"*35)
    
    # 테스트 결과 추적
    test_results = {}
    
    # 각 테스트 실행
    # 테스트는 독립적으로 실행되므로 하나가 실패해도 다음 테스트를 계속 진행합니다.
    
    test_results["연결 테스트"] = await test_mcp_connection()
    
    test_results["도구 가져오기"] = await test_get_tools()
    
    test_results["에이전트 통합"] = await test_agent_with_mcp()
    
    test_results["오류 처리"] = await test_error_handling()
    
    # 최종 결과 출력
    print("\n" + "="*70)
    print("📊 테스트 결과 요약")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n총 {len(test_results)}개 테스트 중:")
    print(f"  ✅ 통과: {passed}개")
    print(f"  ❌ 실패: {failed}개")
    
    # 모든 테스트 통과 여부
    if failed == 0:
        print("\n🎉 모든 테스트를 통과했습니다!\n")
        return 0
    else:
        print(f"\n⚠️ {failed}개의 테스트가 실패했습니다.\n")
        return 1


if __name__ == "__main__":
    # asyncio로 메인 함수 실행
    # 반환값은 exit code (0: 성공, 1: 실패)
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 사용자가 테스트를 중단했습니다.\n")
        sys.exit(1)
