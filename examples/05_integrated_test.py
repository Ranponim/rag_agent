# -*- coding: utf-8 -*-
"""
05. Integrated Test - 모든 기법을 통합한 최종 예제

이 예제는 지금까지 학습한 모든 LangGraph 기법을 통합하여
실전 수준의 RAG Agent 시스템을 구현합니다.

통합된 기법:
    1. Multi-Agent (Supervisor 패턴)
    2. Memory (대화 기록 유지)
    3. Adaptive RAG (쿼리 복잡도 분류)
    4. Tool Calling (외부 도구 활용)
    5. Document Grading (문서 관련성 평가)
    6. Query Transform (쿼리 변환)

실행: python examples/05_integrated_test.py
"""

import sys
from pathlib import Path
from typing import TypedDict, List, Literal, Annotated

# 프로젝트 루트를 경로에 추가하여 내부 모듈(config, utils)을 불러올 수 있게 함
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangChain: 메시지 구조, 도구 정의 및 RAG 관련
from langchain_core.documents import Document  # 표준 문서 객체
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage  # 다양한 메시지 타입
from langchain_core.prompts import ChatPromptTemplate  # 프롬프트 설계도
from langchain_core.tools import tool  # 도구 정의 데코레이터

# LangGraph: 워크플로우 제어, 상태 관리 및 체크포인트
from langgraph.graph import StateGraph, START, END  # 그래프 빌더 및 주요 제어 포인트
from langgraph.graph.message import add_messages  # 메시지 자동 병합 리듀서
from langgraph.prebuilt import ToolNode  # 표준 도구 실행 노드
from langgraph.checkpoint.memory import MemorySaver  # 대화 기록 영속성 관리를 위한 체크포인터

# 프로젝트 유틸리티
from config.settings import get_settings  # 설정 및 환경 변수 로드
from utils.llm_factory import get_llm, get_embeddings  # LLM/임베딩 팩토리
from utils.vector_store import VectorStoreManager  # 벡터 DB 검색 매니저


# =============================================================================
# 1. 통합 State 정의
# =============================================================================

class IntegratedAgentState(TypedDict):
    """통합 Agent 시스템의 상태"""
    # 메시지 히스토리 (Memory)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 쿼리 분석
    current_query: str
    query_type: str                      # "chat" | "search" | "tool"
    query_complexity: str                # "simple" | "moderate" | "complex"
    
    # RAG 관련
    transformed_query: str               # 변환된 쿼리
    documents: List[Document]            # 검색된 문서
    graded_documents: List[Document]     # 평가된 문서
    context: str
    
    # 실행 추적
    current_agent: str
    steps_taken: List[str]


# =============================================================================
# 2. Vector Store 초기화
# =============================================================================

_integrated_vs: VectorStoreManager = None

def get_integrated_vs() -> VectorStoreManager:
    global _integrated_vs
    if _integrated_vs is None:
        print("📚 통합 시스템 Vector Store 초기화...")
        _integrated_vs = VectorStoreManager(
            embeddings=get_embeddings(),
            collection_name="integrated_system",
        )
        samples = [
            "LangGraph는 상태 기반 에이전트를 구축하기 위한 프레임워크입니다. StateGraph로 노드와 엣지를 정의합니다.",
            "RAG(Retrieval-Augmented Generation)는 검색 증강 생성으로, LLM에 외부 지식을 제공합니다.",
            "Multi-Agent 시스템은 여러 전문 Agent가 협력하여 복잡한 작업을 수행합니다.",
            "MemorySaver는 LangGraph에서 대화 기록을 저장하고 복원하는 체크포인터입니다.",
            "Adaptive RAG는 쿼리 복잡도에 따라 다른 RAG 전략을 선택합니다.",
            "Tool Calling은 LLM이 외부 도구를 호출하여 실시간 정보를 얻는 기법입니다.",
            "Document Grading은 검색된 문서의 관련성을 평가하여 품질을 보장합니다.",
            "Query Transform은 원본 쿼리를 변환하여 검색 효율을 높입니다. HyDE, Multi-Query 등이 있습니다.",
        ]
        _integrated_vs.add_texts(texts=samples)
        print(f"✅ {len(samples)}개 문서 추가")
    return _integrated_vs


# =============================================================================
# 3. 도구 정의
# =============================================================================

@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    from datetime import datetime
    return f"현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다."""
    try:
        return f"{expression} = {eval(expression)}"
    except:
        return "계산 오류"


@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다. (데모용)"""
    return f"'{query}' 검색 결과: 관련 정보를 찾았습니다."


tools = [get_current_time, calculate, search_web]


# =============================================================================
# 4. 노드 함수들
# =============================================================================

def router_node(state: IntegratedAgentState) -> dict:
    """
    Router: 쿼리를 분석하여 적절한 처리 경로 결정
    
    - chat: 일반 대화 → 직접 응답
    - search: 정보 검색 필요 → RAG 파이프라인
    - tool: 도구 사용 필요 → Tool Agent
    """
    print("\n🔀 [Router] 쿼리 분석 중...")
    
    # 마지막 사용자 메시지 추출
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """쿼리를 분석하여 처리 방식을 결정하세요.

- chat: 인사, 잡담, 이전 대화 참조 등
- search: 정보 검색이 필요한 질문 (LangGraph, RAG 등)
- tool: 계산, 현재 시간, 웹 검색 등 도구가 필요한 경우

"chat", "search", "tool" 중 하나만 답하세요."""),
        ("human", "쿼리: {query}"),
    ])
    
    response = (prompt | llm).invoke({"query": query})
    content = response.content.lower().strip()
    
    if "tool" in content:
        query_type = "tool"
    elif "search" in content:
        query_type = "search"
    else:
        query_type = "chat"
    
    print(f"   → 쿼리 유형: {query_type}")
    
    return {
        "current_query": query,
        "query_type": query_type,
        "steps_taken": state.get("steps_taken", []) + ["router"]
    }


def chat_node(state: IntegratedAgentState) -> dict:
    """일반 대화 처리"""
    print("\n💬 [Chat] 대화 응답 생성...")
    
    llm = get_llm()
    
    # 이전 대화 컨텍스트 포함
    messages = [
        SystemMessage(content="당신은 친절한 AI 어시스턴트입니다. 이전 대화를 참고하여 자연스럽게 대화하세요.")
    ] + state["messages"]
    
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "steps_taken": state.get("steps_taken", []) + ["chat"]
    }


def query_transform_node(state: IntegratedAgentState) -> dict:
    """쿼리 변환 (HyDE 스타일)"""
    print("\n🔄 [Query Transform] 쿼리 변환 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "질문에 대한 이상적인 답변을 작성하세요. 이 답변은 검색에 사용됩니다."),
        ("human", "{query}"),
    ])
    
    response = (prompt | llm).invoke({"query": state["current_query"]})
    transformed = response.content
    
    print(f"   → 변환된 쿼리: {transformed[:80]}...")
    
    return {
        "transformed_query": transformed,
        "steps_taken": state.get("steps_taken", []) + ["query_transform"]
    }


def retrieve_node(state: IntegratedAgentState) -> dict:
    """문서 검색"""
    print("\n🔍 [Retrieve] 문서 검색 중...")
    
    vs = get_integrated_vs()
    
    # 변환된 쿼리 또는 원본 쿼리 사용
    search_query = state.get("transformed_query") or state["current_query"]
    docs = vs.search(query=search_query, k=5)
    
    print(f"   → {len(docs)}개 문서 검색됨")
    
    return {
        "documents": docs,
        "steps_taken": state.get("steps_taken", []) + ["retrieve"]
    }


def grade_documents_node(state: IntegratedAgentState) -> dict:
    """문서 관련성 평가"""
    print("\n📊 [Grade] 문서 관련성 평가...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "문서가 질문과 관련있으면 'yes', 없으면 'no'만 답하세요."),
        ("human", "질문: {query}\n\n문서: {document}"),
    ])
    
    graded = []
    for i, doc in enumerate(state["documents"]):
        response = (prompt | llm).invoke({
            "query": state["current_query"],
            "document": doc.page_content[:500]
        })
        
        if "yes" in response.content.lower():
            graded.append(doc)
            print(f"   [{i+1}] ✅ 관련 있음")
        else:
            print(f"   [{i+1}] ❌ 관련 없음")
    
    context = "\n\n".join([doc.page_content for doc in graded[:3]])
    
    print(f"   → 관련 문서: {len(graded)}개")
    
    return {
        "graded_documents": graded,
        "context": context,
        "steps_taken": state.get("steps_taken", []) + ["grade"]
    }


def generate_node(state: IntegratedAgentState) -> dict:
    """RAG 답변 생성"""
    print("\n💭 [Generate] 답변 생성 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """컨텍스트를 기반으로 질문에 답변하세요.

컨텍스트:
{context}"""),
        ("human", "{query}"),
    ])
    
    response = (prompt | llm).invoke({
        "context": state.get("context", "정보 없음"),
        "query": state["current_query"]
    })
    
    return {
        "messages": [AIMessage(content=response.content)],
        "steps_taken": state.get("steps_taken", []) + ["generate"]
    }


def tool_agent_node(state: IntegratedAgentState) -> dict:
    """도구 사용 Agent"""
    print("\n🔧 [Tool Agent] 도구 호출 중...")
    
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        SystemMessage(content="필요한 도구를 사용하여 질문에 답하세요.")
    ] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "steps_taken": state.get("steps_taken", []) + ["tool_agent"]
    }


def tool_executor_node(state: IntegratedAgentState) -> dict:
    """도구 실행"""
    print("\n⚙️ [Tool Executor] 도구 실행...")
    
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)
    
    return {
        "messages": result.get("messages", []),
        "steps_taken": state.get("steps_taken", []) + ["tool_executor"]
    }


def should_use_tools(state: IntegratedAgentState) -> Literal["tools", "end"]:
    """도구 호출 여부 확인"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# =============================================================================
# 5. 라우터 함수
# =============================================================================

def route_by_query_type(state: IntegratedAgentState) -> Literal["chat", "rag", "tool"]:
    """쿼리 유형에 따라 라우팅"""
    query_type = state.get("query_type", "chat")
    
    if query_type == "search":
        return "rag"
    elif query_type == "tool":
        return "tool"
    return "chat"


# =============================================================================
# 6. 그래프 생성
# =============================================================================

def create_integrated_agent():
    """
    통합 Agent 그래프 생성
    
    구조:
        START → router → (chat | rag | tool)
        
        chat → END
        
        rag: query_transform → retrieve → grade → generate → END
        
        tool: tool_agent → (tools → tool_agent) | END
    """
    graph = StateGraph(IntegratedAgentState)
    
    # 노드 추가
    graph.add_node("router", router_node)
    graph.add_node("chat", chat_node)
    graph.add_node("query_transform", query_transform_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("tools", tool_executor_node)
    
    # 시작 → 라우터
    graph.add_edge(START, "router")
    
    # 라우터 분기
    graph.add_conditional_edges(
        "router",
        route_by_query_type,
        {
            "chat": "chat",
            "rag": "query_transform",
            "tool": "tool_agent"
        }
    )
    
    # Chat 경로
    graph.add_edge("chat", END)
    
    # RAG 경로
    graph.add_edge("query_transform", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "generate")
    graph.add_edge("generate", END)
    
    # Tool 경로
    graph.add_conditional_edges(
        "tool_agent",
        should_use_tools,
        {
            "tools": "tools",
            "end": END
        }
    )
    graph.add_edge("tools", "tool_agent")
    
    # 메모리 활성화
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    
    print("✅ 통합 Agent 시스템 컴파일 완료!")
    return compiled


# =============================================================================
# 7. 실행 인터페이스
# =============================================================================

def chat_with_agent(graph, thread_id: str, message: str) -> str:
    """Agent와 대화"""
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n{'='*60}")
    print(f"🙋 [{thread_id}] 사용자: {message}")
    print('='*60)
    
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "current_query": "",
            "query_type": "",
            "query_complexity": "",
            "transformed_query": "",
            "documents": [],
            "graded_documents": [],
            "context": "",
            "current_agent": "",
            "steps_taken": []
        },
        config=config
    )
    
    # 실행 경로 출력
    steps = result.get("steps_taken", [])
    print(f"\n📍 실행 경로: {' → '.join(steps)}")
    
    # 최종 응답
    final_message = result["messages"][-1]
    response = final_message.content if hasattr(final_message, "content") else str(final_message)
    
    print(f"\n🤖 [{thread_id}] Agent: {response}")
    print('='*60)
    
    return response


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    from utils.llm_factory import log_llm_error
    
    print("\n" + "="*60)
    print("🚀 통합 테스트 - 모든 기법 결합")
    print("="*60)
    
    try:
        graph = create_integrated_agent()
        
        # 테스트 시나리오
        print("\n📌 시나리오: 다양한 유형의 질문")
        
        # 1. 일반 대화
        chat_with_agent(graph, "test-session", "안녕하세요!")
        
        # 2. 정보 검색 (RAG)
        chat_with_agent(graph, "test-session", "LangGraph가 뭐야?")
        
        # 3. 도구 사용
        chat_with_agent(graph, "test-session", "지금 몇 시야?")
        
        # 4. 계산
        chat_with_agent(graph, "test-session", "123 * 456 계산해줘")
        
        # 5. 이전 대화 참조 (Memory)
        chat_with_agent(graph, "test-session", "아까 LangGraph에 대해 뭐라고 했지?")
        
        print("\n" + "="*60)
        print("✅ 통합 테스트 완료!")
        print("="*60)
        
    except Exception as e:
        log_llm_error(e)
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
