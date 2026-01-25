# -*- coding: utf-8 -*-
"""
LangGraph 표준 구조 베이스 코드 (Standard StateGraph Structure)

이 예제는 LangGraph의 핵심 구성 요소인 State, Nodes, Edges를 명시적으로 정의하여
에이전트의 동작 흐름을 완벽하게 제어하는 정석적인 구조를 보여줍니다.

학습 목표:
1. TypedDict를 이용한 상태(State) 설계
2. 노드(Node) 함수 구현 및 등록
3. 조건부 엣지(Conditional Edges)를 통한 흐름 제어
"""

import sys
from pathlib import Path
from typing import Annotated, TypedDict

# LangGraph 핵심 컴포넌트
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# LangChain 컴포넌트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool

# 프로젝트 설정 로드 (API Key, Base URL 등)
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import get_settings

# 1. 상태 정의 (State Definition)
# 에이전트가 대화 내내 유지하고 업데이트할 데이터 구조를 정의합니다.
class AgentState(TypedDict):
    # add_messages는 새로운 메시지가 들어오면 기존 리스트에 자동으로 추가(append)해줍니다.
    messages: Annotated[list[BaseMessage], add_messages]

# 2. 도구 정의 (Tool Definition)
@tool
def get_weather(city: str) -> str:
    """
    특정 도시의 날씨 정보를 반환합니다.
    
    Args:
        city: 날씨를 확인할 도시 이름 (예: "서울", "부산")
    """
    # 실제 서비스에서는 날씨 API를 호출하겠지만, 여기서는 더미 데이터 사용
    weather_data = {
        "서울": "맑음, 15°C",
        "부산": "흐림, 18°C",
        "제주": "비, 20°C",
        "인천": "맑음, 14°C",
    }
    # 도시 이름으로 날씨 검색, 없으면 안내 메시지 반환
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")

@tool
def calculate(expression: str) -> str:
    """
    수학 표현식을 계산합니다.
    
    Args:
        expression: 계산할 수학 표현식 (예: "2 + 3 * 4", "100 / 5")
    """
    try:
        # 문자열 수식을 실행하여 결과 계산
        result = eval(expression)
        return f"결과: {result}"
    except Exception as e:
        # 계산 중 오류 발생 시 메시지 반환
        return f"계산 오류: {str(e)}"

# 그래프에서 사용할 도구 리스트
tools = [get_weather, calculate]

# 3. 노드 함수 정의 (Node Functions)
def call_model(state: AgentState):
    """
    LLM을 호출하여 다음 행동을 결정하는 노드입니다.
    """
    settings = get_settings()
    
    # 모델 초기화 및 도구 바인딩
    model = ChatOpenAI(
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key,
        model=settings.openai_model
    ).bind_tools(tools)
    
    # 모델 호출
    response = model.invoke(state["messages"])
    
    # 상태 업데이트 결과 반환
    return {"messages": [response]}

# 4. 그래프 구성 (Graph Construction)
def create_graph():
    # 상태를 기반으로 그래프 빌더 초기화
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("agent", call_model)          # AI 모델 호출 노드
    workflow.add_node("tools", ToolNode(tools))      # 도구 실행 노드

    # 엣지 연결 (흐름 설계)
    # 시작점 -> agent 노드
    workflow.add_edge(START, "agent")

    # agent 노드 이후의 조건부 경로 설정
    workflow.add_conditional_edges(
        "agent",
        # tools_condition은 모델의 응답에 'tool_calls'가 있으면 "tools"로, 없으면 END로 보내줍니다.
        tools_condition,
    )

    # 도구 실행이 끝나면 다시 agent에게 돌아가서 최종 답변을 생성하게 함
    workflow.add_edge("tools", "agent")

    # 그래프 컴파일 (실행 가능한 앱으로 변환)
    return workflow.compile()

# 5. 실행부
if __name__ == "__main__":
    app = create_graph()
    
    # 테스트 입력 (날씨 정보와 계산 요청)
    inputs = {"messages": [HumanMessage(content="부산의 날씨를 알려주고, 120 / 5 결과도 알려줘.")]}
    
    # invoke 실행 (최종 결과 출력)
    print("--- 에이전트 실행 시작 ---")
    try:
        # invoke는 모든 노드 실행이 완료된 후 최종 상태(result)를 반환합니다.
        result = app.invoke(inputs)
        
        # 마지막 응답 메시지만 깔끔하게 출력합니다.
        if "messages" in result:
            result["messages"][-1].pretty_print()
            
    except Exception as e:
        print(f"\n[오류 발생] 실행 중 문제가 발생했습니다: {e}")
        print("팁: 로컬 LLM 서버(LM Studio 등)의 연결 상태를 확인해주세요.")
        
    print("--- 에이전트 실행 종료 ---")
