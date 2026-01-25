# -*- coding: utf-8 -*-
"""
============================================================================
📚 05. Integrated Test - 모든 기법을 통합한 최종 완성형 Agent
============================================================================

지금까지 개별적으로 학습한 모든 LangGraph 및 RAG 기법을 하나의 대규모 시스템으로 
통합합니다. 실전에서 사용 가능한 수준의 복합 에이전트 구조를 학습합니다.

🎯 통합된 핵심 기술:
    1. Router (Adaptive): 질문 유형(대화/검색/도구) 및 복잡도 자동 판별
    2. Multi-Agent (Supervisor): 전문 에이전트들에게 작업 분배
    3. Memory (MemorySaver): 세션별 대화 기록 유지 및 문맥 파악
    4. Tool Calling (ReAct): 필요 시 계산기, 시간 조회 등 외부 도구 활용
    5. Advanced RAG (Query Transform & Grading): 쿼리 변환 및 문서 품질 검증
    6. Harmony Support: GPT-OSS(vLLM) 로컬 서버 호환성 완벽 지원

그래프 구조:
    START → router (판별) ─┬→ chat (일반 대화) ──────────────→ END
                            ├→ rag_flow (검색/평가/생성) ─────→ END
                            └→ tool_agent (도구/실행) ──🔁───→ END

실행 방법:
    python examples/05_integrated_test.py
"""

# =============================================================================
# 📦 필수 라이브러리 임포트
# =============================================================================

import sys                              # 시스템 경로 조작
from pathlib import Path                # 경로 관리
from typing import TypedDict, List, Literal, Annotated  # 타입 힌팅

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangChain 구성 요소
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# LangGraph 구성 요소
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  # 메시지를 덮어쓰지 않고 추가(Append)하는 리듀서
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 프로젝트 유틸리티
from config.settings import get_settings
from utils.llm_factory import get_llm, get_embeddings, log_llm_error
from utils.vector_store import VectorStoreManager


# =============================================================================
# 📋 1. 통합 상태(State) 정의
# =============================================================================

class IntegratedState(TypedDict):
    """시스템 전체를 관통하는 통합 상태 딕셔너리"""
    # 💡 Annotated와 add_messages를 사용하여 메시지 리스트가 자동으로 누적되게 함
    messages: Annotated[List[BaseMessage], add_messages]
    
    current_query: str                # 사용자의 최근 질문
    query_type: str                   # 질문 유형 ("chat", "rag", "tool")
    
    # RAG 관련 필드
    transformed_query: str            # 검색용으로 변환된 질문
    context: str                      # 검색 및 검증된 문맥 데이터
    
    # 추적용 필드
    steps_taken: List[str]            # 어떤 노드를 거쳐왔는지 기록 (디버깅용)


# =============================================================================
# 🗄️ 2. Vector Store & Tools 준비
# =============================================================================

def get_combined_vs() -> VectorStoreManager:
    """통합 테스트용 지식 데이터 로드"""
    embeddings = get_embeddings()
    manager = VectorStoreManager(embeddings=embeddings, collection_name="integrated_final")
    if True:
        samples = [
            "LangGraph는 순환 그래프를 지원하는 에이전트 개발 프레임워크입니다.",
            "MemorySaver를 쓰면 thread_id별로 대화 내용을 기억할 수 있습니다.",
            "Reranking은 검색된 문서의 우선순위를 LLM이 다시 매기는 기술입니다.",
            "HyDE는 가짜 답변을 생성해 검색 정확도를 높이는 쿼리 변형 기법입니다.",
            "에이전트는 LLM이 도구 사용 여부를 스스로 결정하는 시스템을 말합니다.",
        ]
        manager.add_texts(samples)
    return manager

@tool
def calculate_math(expression: str) -> str:
    """복잡한 수학 계산을 수행합니다."""
    try: return f"결과: {eval(expression)}"
    except: return "계산할 수 없는 수식입니다."

@tool
def get_system_time() -> str:
    """현재 시스템의 날짜와 시간을 확인합니다."""
    from datetime import datetime
    return f"현재 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# 그래프에서 사용할 도구 리스트
tools = [calculate_math, get_system_time]


# =============================================================================
# 🧠 3. 노드 함수 정의 (노드별 전문 역할)
# =============================================================================

def router_node(state: IntegratedState) -> dict:
    """
    [노드 1] 라우터: 질문의 의도를 파악하여 경로를 배정합니다.
    """
    print("\n🧐 [Router] 사용자 질문 분석 중...")
    last_msg = state["messages"][-1].content
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "질문을 분석하여 'chat'(단순대화), 'rag'(지식검색), 'tool'(도구사용) 중 하나로 분류하세요. 단어 하나만 답하세요."),
        ("human", "{query}"),
    ])
    
    res = (prompt | llm).invoke({"query": last_msg})
    q_type = res.content.lower().strip()
    
    # 안전 장치: 분류 실패 시 기본 chat
    if q_type not in ["chat", "rag", "tool"]: q_type = "chat"
    
    print(f"   → 분석 결과: '{q_type}' 경로로 배정")
    return {"query_type": q_type, "current_query": last_msg, "steps_taken": ["router"]}


def chat_node(state: IntegratedState) -> dict:
    """
    [노드 2] 일반 대화: 대화 지침을 기반으로 친절하게 답변합니다.
    """
    print("💬 [Chat] 일상 대화 또는 가벼운 응답 생성 중...")
    llm = get_llm()
    # 시스템 지침과 대화 메시지를 합쳐서 AI에게 전달합니다.
    messages = [SystemMessage(content="당신은 다정하고 똑똑한 비서입니다.")] + state["messages"]
    res = llm.invoke(messages)
    return {"messages": [res], "steps_taken": state["steps_taken"] + ["chat"]}


def rag_pipeline_node(state: IntegratedState) -> dict:
    """
    [노드 3] 통합 RAG: 쿼리 변환, 검색, 문서 평가를 한 번에 처리합니다.
    (복잡성을 줄이기 위해 하나의 노드에서 처리하거나, 원하면 더 나눌 수 있습니다)
    """
    print("🔍 [RAG] 지식 검색 및 문서 검증 진행 중...")
    llm = get_llm()
    
    # 1. 쿼리 변환 (HyDE)
    hyde_res = llm.invoke(f"질문: {state['current_query']}\n이 질문에 대한 가상의 짧은 답변을 작성해 주세요.")
    
    # 2. 검색
    vs = get_combined_vs()
    docs = vs.search(hyde_res.content, k=3)
    
    # 3. 문서 평가 (Grading)
    valid_docs = []
    for d in docs:
        grade = llm.invoke(f"문서: {d.page_content}\n질문: {state['current_query']}\n관련 있으면 'yes' 없으면 'no'라고만 하세요.")
        if "yes" in grade.content.lower():
            valid_docs.append(d.page_content)
    
    context = "\n".join(valid_docs) if valid_docs else "관련 정보를 찾지 못했습니다."
    
    # 4. 답변 생성
    ans = llm.invoke(f"참조:\n{context}\n\n질문: {state['current_query']}\n답변해 주세요.")
    
    return {"messages": [ans], "steps_taken": state["steps_taken"] + ["integrated_rag"]}


def tool_agent_node(state: IntegratedState) -> dict:
    """
    [노드 4] 도구 에이전트: 도구를 선택하고 사용합니다.
    """
    print("🔧 [Tool Agent] 필요한 도구 탐색 및 실행 결정 중...")
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    
    # AI에게 메시지를 전달하고 도구 호출 응답을 받습니다.
    res = llm_with_tools.invoke(state["messages"])
    
    return {"messages": [res], "steps_taken": state["steps_taken"] + ["tool_agent"]}


# =============================================================================
# 🚦 4. 라우터 및 조건부 로직
# =============================================================================

def route_selection(state: IntegratedState) -> Literal["chat", "rag", "tool"]:
    """라우터 노드 이후 어디로 갈지 결정"""
    return state["query_type"]

def check_further_tools(state: IntegratedState) -> Literal["tools", "end"]:
    """도구를 더 써야 하는지 판단 (ReAct 루프)"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        print(f"   → 실행할 도구 발견: {[tc['name'] for tc in last_msg.tool_calls]}")
        return "tools"
    return "end"


# =============================================================================
# 🔗 5. 그래프 조립 (Complete Graph)
# =============================================================================

def create_integrated_system():
    """모든 노드와 엣지를 연결하여 완성된 시스템을 만듭니다."""
    builder = StateGraph(IntegratedState)
    
    # 노드 등록
    builder.add_node("router", router_node)
    builder.add_node("chat", chat_node)
    builder.add_node("rag", rag_pipeline_node)
    builder.add_node("tool_agent", tool_agent_node)
    builder.add_node("tools", ToolNode(tools))  # 실제 도구를 실행하는 prebuilt 노드
    
    # 엣지 연결
    builder.add_edge(START, "router")
    
    # 라우터에서의 분기
    builder.add_conditional_edges(
        "router", 
        route_selection, 
        {"chat": "chat", "rag": "rag", "tool": "tool_agent"}
    )
    
    # Chat과 RAG는 완료 후 종료
    builder.add_edge("chat", END)
    builder.add_edge("rag", END)
    
    # 도구 에이전트는 루프 구조 (ReAct)
    builder.add_conditional_edges(
        "tool_agent", 
        check_further_tools, 
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "tool_agent") # 도구 실행 후 다시 에이전트로 가서 결과 요약
    
    # 💾 대화 기록 유지를 위한 메모리 체크포인터
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# =============================================================================
# ▶️ 6. 실행 및 인터페이스 (CLI)
# =============================================================================

def run_chat_loop(graph, thread_id: str):
    """지속적인 대화를 위한 CLI 루프"""
    print("\n" + "="*60)
    print("🚀 통합 AI 에이전트 시스템 가동 중...")
    print(f"현재 세션 ID: {thread_id}")
    print("="*60)
    print("- 'quit' 또는 'exit'를 입력하여 종료")
    print("- 아무 질문이나 던져보세요 (대화, 기술 질문, 계산 등)")
    print("="*60)

    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            user_input = input("\n🙋 사용자: ").strip()
            if not user_input: continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 시스템을 종료합니다. 안녕히 가세요!")
                break
                
            # 그래프 실행
            # 💡 messages에 내용을 담아 넘기면 Annotated 리듀서에 의해 자동 추가됨
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config
            )
            
            # 최종 응답 출력
            ans = result["messages"][-1].content
            print(f"\n🤖 Agent: {ans}")
            
            # 디버깅 정보 (어떤 과정을 거쳤나?)
            path = " → ".join(result.get("steps_taken", []))
            print(f"💡 [실행 경로: {path}]")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            log_llm_error(e)
            print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    # 1. 시스템 초기화 (그래프 생성)
    final_agent = create_integrated_system()
    
    # 2. 고유 세션 ID 생성 (또는 고정값 사용)
    my_thread_id = "final_test_user_001"
    
    # 3. CLI 대화 루프 시작
    run_chat_loop(final_agent, my_thread_id)
