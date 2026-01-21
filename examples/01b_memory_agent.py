# -*- coding: utf-8 -*-
"""
01b. Memory Agent - 대화 기록을 유지하는 Agent

이 예제는 MemorySaver를 사용하여 대화 기록을 유지하고,
thread_id로 여러 대화 세션을 관리하는 Agent를 구현합니다.

학습 목표:
    1. MemorySaver 체크포인터 사용법
    2. thread_id로 세션 분리
    3. 대화 컨텍스트 유지
    4. 이전 대화 참조

실행: python examples/01b_memory_agent.py
"""

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from config.settings import get_settings
from utils.llm_factory import get_llm


# =============================================================================
# 1. 도구 정의
# =============================================================================

@tool
def remember_name(name: str) -> str:
    """
    사용자의 이름을 기억합니다.
    
    Args:
        name: 기억할 이름
    
    Returns:
        str: 확인 메시지
    """
    return f"'{name}'님의 이름을 기억했습니다!"


@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


tools = [remember_name, calculate]


# =============================================================================
# 2. Agent 노드
# =============================================================================

def agent_node(state: MessagesState) -> dict:
    """대화 컨텍스트를 유지하는 Agent"""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    system_message = SystemMessage(content="""당신은 친절한 AI 어시스턴트입니다.

중요한 특징:
- 이전 대화 내용을 기억하고 참조할 수 있습니다
- 사용자 이름을 기억하고 적절히 사용합니다
- 대화 흐름에 맞는 자연스러운 응답을 합니다

이전 대화를 참조하여 일관성 있는 대화를 유지하세요.
""")
    
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


# =============================================================================
# 3. 라우터 함수
# =============================================================================

def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"🔧 도구 호출: {[tc['name'] for tc in last_message.tool_calls]}")
        return "tools"
    
    return END


# =============================================================================
# 4. 메모리 기능이 있는 그래프 생성
# =============================================================================

def create_memory_agent():
    """
    메모리 기능이 있는 Agent 그래프 생성
    
    Returns:
        CompiledGraph: 메모리가 활성화된 컴파일된 그래프
    """
    graph = StateGraph(MessagesState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    
    # ⭐ 핵심: MemorySaver로 상태 저장 활성화
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    
    print("✅ Memory Agent 컴파일 완료! (체크포인터 활성화)")
    return compiled


# =============================================================================
# 5. 세션별 대화 실행
# =============================================================================

def chat(graph, thread_id: str, message: str) -> str:
    """
    특정 세션(thread_id)에서 대화를 수행합니다.
    
    Args:
        graph: 컴파일된 그래프
        thread_id: 대화 세션 ID
        message: 사용자 메시지
    
    Returns:
        str: Agent 응답
    """
    # ⭐ config에 thread_id 지정 → 같은 thread_id면 이전 대화 유지
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n💬 [{thread_id}] 사용자: {message}")
    
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    
    response = result["messages"][-1].content
    print(f"🤖 [{thread_id}] Agent: {response}")
    
    return response


def show_conversation_history(graph, thread_id: str):
    """특정 세션의 대화 기록을 표시합니다."""
    config = {"configurable": {"thread_id": thread_id}}
    
    # 현재 상태 스냅샷 조회
    state = graph.get_state(config)
    
    print(f"\n📜 [{thread_id}] 대화 기록:")
    print("-" * 40)
    
    if state.values and "messages" in state.values:
        for msg in state.values["messages"]:
            msg_type = type(msg).__name__
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"  [{msg_type}] {content}")
    else:
        print("  (기록 없음)")
    
    print("-" * 40)


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    from utils.llm_factory import log_llm_error
    
    print("\n" + "="*60)
    print("Memory Agent 예제 - 대화 기록 유지")
    print("="*60)
    
    try:
        graph = create_memory_agent()
        
        # ====================================
        # 시나리오 1: User A와의 대화 (thread-A)
        # ====================================
        print("\n" + "="*60)
        print("📌 시나리오 1: User A와의 대화")
        print("="*60)
        
        chat(graph, "thread-A", "안녕! 내 이름은 철수야.")
        chat(graph, "thread-A", "내 이름이 뭐라고 했지?")  # → 이전 대화 참조
        chat(graph, "thread-A", "10 + 20 계산해줘")
        
        # ====================================
        # 시나리오 2: User B와의 대화 (thread-B)
        # ====================================
        print("\n" + "="*60)
        print("📌 시나리오 2: User B와의 대화 (별도 세션)")
        print("="*60)
        
        chat(graph, "thread-B", "안녕하세요, 저는 영희입니다.")
        chat(graph, "thread-B", "제 이름 기억하세요?")  # → thread-B의 대화만 참조
        
        # ====================================
        # 시나리오 3: 다시 User A와 대화 (이전 기록 유지)
        # ====================================
        print("\n" + "="*60)
        print("📌 시나리오 3: 다시 User A (이전 대화 기억)")
        print("="*60)
        
        chat(graph, "thread-A", "아까 계산 결과가 뭐였지?")  # → thread-A의 이전 대화 참조
        
        # ====================================
        # 대화 기록 확인
        # ====================================
        print("\n" + "="*60)
        print("📌 대화 기록 확인")
        print("="*60)
        
        show_conversation_history(graph, "thread-A")
        show_conversation_history(graph, "thread-B")
        
    except Exception as e:
        log_llm_error(e)
        print(f"❌ 오류: {e}")
