# -*- coding: utf-8 -*-
"""
============================================================================
📚 02c. RAG Tool Agent - create_react_agent를 활용한 RAG 에이전트
============================================================================

LangGraph의 `create_react_agent`를 사용하여 RAG 검색을 Tool로 활용하는
에이전트를 구현합니다. Agent가 질문의 맥락에 따라 자동으로 문서 검색 여부를
판단하여 더 유연한 대화가 가능합니다.

🎯 학습 목표:
    1. RAG 검색을 Tool로 래핑하는 방법
    2. create_react_agent를 사용한 간결한 에이전트 구성
    3. Agent가 Tool 호출 여부를 자동으로 판단하는 ReAct 패턴

💡 기존 RAG (02_naive_rag.py)와의 차이점:
    - 02_naive_rag: StateGraph로 수동 그래프 구성, 항상 검색 수행
    - 02c_rag_tool_agent: create_react_agent 자동 그래프, 필요시에만 검색

그래프 구조:
    START → Agent → (Tool 호출 필요?) → RAG Search Tool → Agent → END
    
    Agent가 Tool 호출이 필요하다고 판단하면 search_documents를 호출하고,
    그 결과를 바탕으로 최종 답변을 생성합니다.

실행 방법:
    python examples/02c_rag_tool_agent.py
    
    실행 후 CLI에서 질문을 입력하면 Agent가 필요시 RAG 검색을 수행하여 응답합니다.
    종료: 'quit', 'exit', 또는 'q' 입력
"""

# =============================================================================
# 📦 필수 라이브러리 임포트
# =============================================================================

# Python 표준 라이브러리
import sys                              # 시스템 경로 조작용
from pathlib import Path                # 파일 경로를 객체지향적으로 다루는 라이브러리
import os                               # 환경변수 접근용

# 프로젝트 루트를 Python 경로에 추가하여 내부 모듈 import 가능하게 함
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# 🔗 LangGraph 핵심 모듈 임포트
# -----------------------------------------------------------------------------

from langgraph.prebuilt import create_react_agent
# create_react_agent: LangGraph에서 제공하는 프리빌트 ReAct 에이전트 생성기
# - 자동으로 Tool 호출 여부를 판단하고 실행하는 그래프를 생성
# - 수동으로 StateGraph를 구성할 필요 없이 한 줄로 에이전트 생성 가능

# -----------------------------------------------------------------------------
# 🔗 LangChain 핵심 모듈 임포트
# -----------------------------------------------------------------------------

from langchain_openai import ChatOpenAI
# ChatOpenAI: OpenAI API(또는 로컬 LLM)와 통신하는 LangChain 클래스

from langchain_core.messages import HumanMessage
# HumanMessage: 사용자 메시지를 나타내는 클래스

from langchain_core.tools import tool
# tool: 함수를 LangChain/LangGraph Tool로 변환하는 데코레이터
# Agent가 이 Tool을 호출할 수 있게 됨

# -----------------------------------------------------------------------------
# 🔗 프로젝트 내부 유틸리티 임포트
# -----------------------------------------------------------------------------

from utils.data_loader import get_rag_vector_store
# get_rag_vector_store: ./rag 폴더의 파일을 자동으로 로드하고 벡터화하는 함수
# 이미 벡터화된 데이터가 있으면 재사용, 변경이 있으면 자동 재임베딩


# =============================================================================
# 🔧 1. RAG Search Tool 정의
# =============================================================================
#
# @tool 데코레이터를 사용하여 함수를 Agent가 호출할 수 있는 Tool로 변환합니다.
# Tool의 docstring은 Agent가 이 Tool을 사용할지 결정하는 데 중요한 역할을 합니다.
# =============================================================================

@tool
def search_documents(query: str) -> str:
    """
    주어진 쿼리와 관련된 문서를 검색하여 컨텍스트를 제공합니다.
    
    RAG 지식 창고에서 질문과 관련된 문서를 검색합니다.
    LangGraph, RAG, AI 에이전트 등에 대한 정보가 필요할 때 사용하세요.
    
    Args:
        query: 검색할 질문 또는 키워드
        
    Returns:
        str: 검색된 문서들의 내용을 합친 컨텍스트 문자열
    """
    # 로그 출력: Tool이 호출되었음을 사용자에게 알림
    print(f"\n🔎 [RAG Tool 호출] 검색 쿼리: {query}")
    
    # Vector Store 가져오기
    # - collection_name: 사용할 벡터 컬렉션 이름
    # - ./rag 폴더의 파일을 자동으로 로드하고 벡터화
    vs = get_rag_vector_store(collection_name="rag_tool_collection")
    
    # 유사도 검색 수행
    # - k: 검색할 문서 개수 (상위 3개)
    docs = vs.search(query, k=3)
    
    # 검색 결과 로깅
    print(f"   → {len(docs)}개 문서 검색됨")
    for i, doc in enumerate(docs):
        # 각 문서의 처음 50자를 미리보기로 출력
        preview = doc.page_content[:50].replace('\n', ' ')
        print(f"   [{i+1}] {preview}...")
    
    # 검색된 문서들의 내용을 하나의 문자열로 합침
    # Agent가 이 컨텍스트를 바탕으로 답변을 생성
    if docs:
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return context
    else:
        return "관련 문서를 찾을 수 없습니다."


# 그래프에서 사용할 도구들을 리스트로 묶어줍니다.
tools = [search_documents]


# =============================================================================
# 🤖 2. Agent 생성 함수
# =============================================================================
#
# create_react_agent를 사용하여 ReAct 패턴의 에이전트를 생성합니다.
# Agent는 자동으로 Tool 호출 여부를 판단하고, 필요시 Tool을 실행합니다.
# =============================================================================

def create_agent():
    """
    RAG Tool을 사용하는 ReAct Agent를 생성합니다.
    
    Returns:
        CompiledGraph: 실행 가능한 컴파일된 에이전트 그래프
        
    💡 create_react_agent가 자동으로 구성하는 그래프 구조:
        1. Agent Node: LLM이 Tool 호출 여부 결정
        2. Tools Node: Tool 실행 (search_documents)
        3. 조건 엣지: tool_calls 여부에 따라 분기
    """
    # -------------------------------------------------------------------------
    # Step 1: LLM 모델 초기화
    # -------------------------------------------------------------------------
    
    # 환경변수에서 LLM 설정을 읽어서 ChatOpenAI 인스턴스 생성
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),    # API 엔드포인트 (로컬 LLM 지원)
        api_key=os.getenv("OPENAI_API_KEY"),      # API 키
        model=os.getenv("OPENAI_MODEL")           # 사용할 모델 이름
    )
    
    # -------------------------------------------------------------------------
    # Step 2: 시스템 프롬프트 정의
    # -------------------------------------------------------------------------
    
    # Agent의 역할과 지침을 정의
    # Tool 사용 방법과 답변 스타일을 명시
    system_prompt = """당신은 RAG(Retrieval-Augmented Generation) 시스템을 사용하는 지식 기반 AI 비서입니다.

사용자의 질문에 답변할 때:
1. 관련 정보가 필요하면 search_documents 도구를 사용하여 문서를 검색하세요.
2. 검색된 문서의 내용을 바탕으로 정확하고 상세한 답변을 생성하세요.
3. 답변에 사용한 정보의 출처가 검색된 문서임을 언급하세요.

모든 답변은 한국어로 친절하고 명확하게 작성하세요.
단순한 인사나 일반적인 질문에는 도구 사용 없이 직접 답변해도 됩니다."""
    
    # -------------------------------------------------------------------------
    # Step 3: ReAct Agent 생성
    # -------------------------------------------------------------------------
    
    # create_react_agent를 사용하여 한 줄로 그래프 생성
    # - model: 사용할 LLM
    # - tools: Agent가 사용할 수 있는 도구 목록
    # - prompt: 시스템 프롬프트 (Agent의 역할 정의)
    app = create_react_agent(
        model,
        tools=tools,
        prompt=system_prompt
    )
    
    return app


# =============================================================================
# ▶️ 3. CLI 실행부
# =============================================================================
#
# 사용자가 터미널에서 직접 Agent와 대화할 수 있는 인터페이스를 제공합니다.
# 기존 01_base_agent_react.py와 동일한 패턴을 사용합니다.
# =============================================================================

if __name__ == "__main__":
    # Agent 생성
    app = create_agent()
    
    # 시작 메시지 출력
    print("\n" + "=" * 60)
    print("🤖 LangGraph RAG Tool Agent (CLI 대화 모드)")
    print("=" * 60)
    print("RAG 검색 도구를 사용하여 질문에 답변합니다.")
    print("필요시 자동으로 문서를 검색하여 답변을 생성합니다.")
    print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.\n")
    
    # CLI 대화 루프
    while True:
        try:
            # -----------------------------------------------------------------
            # 사용자 입력 받기
            # -----------------------------------------------------------------
            user_input = input("👤 You: ").strip()
            
            # 종료 조건 확인
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 대화를 종료합니다. 안녕히 가세요!")
                break
            
            # 빈 입력 처리
            if not user_input:
                print("⚠️  메시지를 입력해주세요.\n")
                continue
            
            # -----------------------------------------------------------------
            # Agent 호출
            # -----------------------------------------------------------------
            
            # 입력 메시지 구성
            # create_react_agent는 messages 리스트를 입력으로 받음
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            # Agent 실행
            # invoke: 동기 실행 (모든 처리가 완료될 때까지 대기)
            result = app.invoke(inputs)
            
            # -----------------------------------------------------------------
            # 응답 출력
            # -----------------------------------------------------------------
            
            if "messages" in result:
                print("\n🤖 Agent: ", end="")
                # 마지막 메시지의 content만 추출하여 출력
                print(result["messages"][-1].content)
            print()  # 줄바꿈
            
        except KeyboardInterrupt:
            # Ctrl+C로 종료 시
            print("\n\n👋 대화를 종료합니다. 안녕히 가세요!")
            break
        except Exception as e:
            # 오류 발생 시 상세 정보 출력
            print(f"\n❌ [오류 발생] {e}")
            print("팁: 로컬 LLM 서버(LM Studio 등)의 연결 상태를 확인해주세요.\n")
