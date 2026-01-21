# -*- coding: utf-8 -*-
"""
01. Basic Agent 예제 - LangGraph 기본 개념 학습

이 예제는 LangGraph의 최신 표준 패턴을 적용한 기본 Agent입니다.
`MessagesState`와 `tools_condition`을 사용하여 간결하고 표준적인 그래프를 구현합니다.

학습 목표:
    1. StateGraph(MessagesState) 표준 구조 이해
    2. prebuilt.tools_condition을 이용한 조건부 분기 표준화
    3. LLM에 도구 바인딩 및 상태 관리

실행 방법:
    python examples/01_basic_agent.py
"""

import sys
from pathlib import Path
from typing import Annotated, Literal

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from config.settings import get_settings
from utils.llm_factory import get_llm, log_llm_error


# =============================================================================
# 1. 도구(Tool) 정의
# =============================================================================

@tool
def get_weather(city: str) -> str:
    """특정 도시의 날씨 정보를 반환합니다."""
    weather_data = {
        "서울": "맑음, 15°C",
        "부산": "흐림, 18°C",
        "제주": "비, 20°C",
        "인천": "맑음, 14°C",
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")


@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다."""
    try:
        return f"결과: {eval(expression)}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


tools = [get_weather, calculate]


# =============================================================================
# 2. Agent 노드 정의
# =============================================================================

def agent_node(state: MessagesState):
    """
    Agent 노드: 메시지 히스토리를 기반으로 LLM 응답 생성
    
    표준 패턴:
    - LLM 인스턴스는 노드 내부 혹은 외부에서 준비 가능
    - bind_tools로 도구 정보 주입
    - state["messages"] 전체를 전달하여 문맥 유지
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    # 시스템 메시지가 필요하다면 맨 앞에 추가 (messages 리스트에는 영향 없음)
    # 실제 구현에서는 state에 system message를 관리하거나 여기서 매번 추가할 수 있음
    sys_msg = SystemMessage(content="당신은 날씨 조회와 계산을 돕는 유용한 어시스턴트입니다.")

    # 메시지 리스트 구성
    messages = [sys_msg] + state["messages"]
    
    # LLM 호출
    response = llm_with_tools.invoke(messages)
    
    # 새로운 메시지만 반환 (MessagesState가 자동으로 append 처리)
    return {"messages": [response]}


# =============================================================================
# 3. 그래프 구성 (표준 패턴)
# =============================================================================

def create_agent_graph():
    """
    LangGraph 표준 패턴을 적용한 Agent 그래프 생성
    
    특징:
    - MessagesState 사용
    - prebuilt.ToolNode 사용
    - prebuilt.tools_condition 사용 (직접 라우터 함수 작성 불필요)
    """
    # 1. 그래프 빌더 초기화
    builder = StateGraph(MessagesState)
    
    # 2. 노드 추가
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    
    # 3. 엣지 추가
    # 시작 -> 에이전트
    builder.add_edge(START, "agent")
    
    # 조건부 엣지 (표준 라우터 사용)
    # tools_condition은:
    # - tool_calls가 있으면 "tools"로 이동
    # - 없으면 END로 이동
    builder.add_conditional_edges(
        "agent",
        tools_condition,
    )
    
    # 도구 실행 후 다시 에이전트로 (ReAct 패턴)
    builder.add_edge("tools", "agent")
    
    # 4. 컴파일
    return builder.compile()


# =============================================================================
# 4. 실행 및 테스트
# =============================================================================

def run_agent(query: str):
    """Agent 실행 함수"""
    graph = create_agent_graph()
    
    print(f"\n{'='*60}")
    print(f"🙋 사용자: {query}")
    print('='*60)
    
    try:
        # 스트리밍 모드로 실행하여 과정 시각화
        events = graph.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="values"
        )
        
        final_msg = None
        for event in events:
            if "messages" in event:
                final_msg = event["messages"][-1]
                # 도구 호출이 아닌 경우에만 출력 (너무 시끄러울 수 있음)
                if not (hasattr(final_msg, "tool_calls") and final_msg.tool_calls):
                    # print(f"🤖 Agent: {final_msg.content}")
                    pass
        
        if final_msg:
             print(f"\n🤖 최종 답변: {final_msg.content}")

    except Exception as e:
        log_llm_error(e)
        print("❌ 실행 중 오류가 발생했습니다.")


if __name__ == "__main__":
    print("\nLangGraph Basic Agent (Standard Pattern)")
    
    queries = [
        "서울 날씨 어때?",
        "25 * 4 계산해줘",
        "안녕하세요",
    ]
    
    for q in queries:
        run_agent(q)
