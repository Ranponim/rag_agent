# -*- coding: utf-8 -*-
"""
01. Basic Agent 예제 - LangGraph 기본 개념 학습

이 예제는 LangGraph의 핵심 개념을 학습하기 위한 기본 Agent입니다.
도구(Tool)를 사용하는 간단한 Agent를 구현하여 LangGraph의 동작 원리를 이해합니다.

학습 목표:
    1. StateGraph의 기본 구조 이해
    2. 노드(Node)와 엣지(Edge) 개념 학습
    3. 도구(Tool) 바인딩 방법 이해
    4. 조건부 분기 구현 방법 학습

실행 방법:
    python examples/01_basic_agent.py

필수 환경 변수:
    OPENAI_API_KEY: OpenAI API 키
"""

import sys
from pathlib import Path
from typing import Annotated, Literal

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from config.settings import get_settings
from utils.llm_factory import get_llm


# =============================================================================
# 1. 도구(Tool) 정의
# =============================================================================
# LangGraph에서 도구는 Agent가 외부 작업을 수행할 때 사용합니다.
# @tool 데코레이터를 사용하여 Python 함수를 도구로 변환합니다.

@tool
def get_weather(city: str) -> str:
    """
    특정 도시의 날씨 정보를 반환합니다.
    
    Args:
        city: 날씨를 조회할 도시명
    
    Returns:
        str: 해당 도시의 날씨 정보
    """
    # 실제로는 외부 API를 호출하지만, 예제에서는 더미 데이터 반환
    weather_data = {
        "서울": "맑음, 15°C",
        "부산": "흐림, 18°C",
        "제주": "비, 20°C",
        "인천": "맑음, 14°C",
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")


@tool
def calculate(expression: str) -> str:
    """
    수학 표현식을 계산합니다.
    
    Args:
        expression: 계산할 수학 표현식 (예: "2 + 2", "10 * 5")
    
    Returns:
        str: 계산 결과
    """
    try:
        # 주의: 실제 프로덕션에서는 eval() 사용을 피해야 합니다
        result = eval(expression)
        return f"결과: {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


# 사용할 도구 목록
tools = [get_weather, calculate]


# =============================================================================
# 2. Agent 노드 정의
# =============================================================================

def agent_node(state: MessagesState):
    """
    Agent 노드: LLM을 호출하여 응답을 생성합니다.
    
    이 노드는 그래프의 핵심으로, 사용자 메시지를 받아 LLM에 전달하고
    LLM의 응답(도구 호출 또는 최종 답변)을 반환합니다.
    
    Args:
        state: 현재 그래프 상태 (MessagesState)
               - messages: 대화 메시지 리스트
    
    Returns:
        dict: 업데이트된 상태 (새 메시지 포함)
    
    Note:
        MessagesState는 add_messages 리듀서를 사용하여
        새 메시지가 기존 리스트에 자동으로 추가됩니다.
    """
    # 설정 로드 및 LLM 생성
    settings = get_settings()
    llm = get_llm()
    
    # 도구를 LLM에 바인딩
    # bind_tools()는 LLM이 어떤 도구를 사용할 수 있는지 알려줍니다
    llm_with_tools = llm.bind_tools(tools)
    
    # 시스템 메시지 추가 (선택사항)
    system_message = SystemMessage(
        content="당신은 친절한 도우미입니다. 날씨 조회와 계산을 도와줄 수 있습니다."
    )
    
    # LLM 호출
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # 응답을 메시지 리스트에 추가
    return {"messages": [response]}


# =============================================================================
# 3. 라우터 함수 정의
# =============================================================================

def should_continue(state: MessagesState) -> Literal["tools", END]:
    """
    다음에 실행할 노드를 결정하는 라우터 함수입니다.
    
    LLM의 응답을 분석하여:
    - 도구 호출이 필요하면 "tools" 노드로 이동
    - 최종 응답이면 END로 이동하여 그래프 종료
    
    Args:
        state: 현재 그래프 상태
    
    Returns:
        str: 다음 노드 이름 ("tools" 또는 END)
    
    Note:
        이 함수는 add_conditional_edges()에서 사용됩니다.
        조건부 엣지는 그래프의 분기 로직을 구현합니다.
    """
    # 마지막 메시지 가져오기
    last_message = state["messages"][-1]
    
    # AIMessage의 tool_calls 속성 확인
    # tool_calls가 있으면 도구 실행이 필요
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"🔧 도구 호출 감지: {[tc['name'] for tc in last_message.tool_calls]}")
        return "tools"
    
    # tool_calls가 없으면 최종 응답
    print("✅ 최종 응답 생성 완료")
    return END


# =============================================================================
# 4. 그래프 구성
# =============================================================================

def create_agent_graph():
    """
    Agent 그래프를 생성하고 컴파일합니다.
    
    그래프 구조:
        START → agent → (조건 분기) → tools → agent → ... → END
    
    Returns:
        CompiledGraph: 컴파일된 실행 가능한 그래프
    
    Note:
        1. StateGraph: 상태를 관리하는 그래프 빌더
        2. add_node(): 노드(작업 단위) 추가
        3. add_edge(): 무조건적 연결 추가
        4. add_conditional_edges(): 조건부 연결 추가
        5. compile(): 그래프를 실행 가능한 형태로 변환
    """
    # StateGraph 생성 (MessagesState 사용)
    # MessagesState는 messages 필드를 가진 기본 상태 타입입니다
    graph = StateGraph(MessagesState)
    
    # ----- 노드 추가 -----
    # add_node(이름, 함수): 그래프에 노드(작업 단위)를 추가합니다
    graph.add_node("agent", agent_node)
    
    # ToolNode: 도구 실행을 위한 특수 노드
    # LLM이 요청한 도구를 자동으로 실행합니다
    tool_node = ToolNode(tools)
    graph.add_node("tools", tool_node)
    
    # ----- 엣지 추가 -----
    # add_edge(시작, 끝): 두 노드를 연결합니다
    # START는 그래프의 시작점을 나타내는 특수 상수입니다
    graph.add_edge(START, "agent")
    
    # ----- 조건부 엣지 추가 -----
    # add_conditional_edges(시작 노드, 라우터 함수)
    # should_continue 함수의 반환값에 따라 다음 노드가 결정됩니다
    graph.add_conditional_edges(
        "agent",           # 시작 노드
        should_continue,   # 라우터 함수
        # 경로 매핑 (선택사항, 반환값과 노드명이 같으면 생략 가능)
        # {"tools": "tools", END: END}
    )
    
    # tools 노드 실행 후 다시 agent 노드로 이동
    # 도구 실행 결과를 바탕으로 LLM이 다시 응답을 생성합니다
    graph.add_edge("tools", "agent")
    
    # ----- 그래프 컴파일 -----
    # compile(): StateGraph를 실행 가능한 CompiledGraph로 변환
    compiled_graph = graph.compile()
    
    print("✅ Agent 그래프 컴파일 완료!")
    return compiled_graph


# =============================================================================
# 5. 그래프 실행
# =============================================================================

def run_agent(query: str):
    """
    Agent를 실행하여 사용자 쿼리에 응답합니다.
    
    Args:
        query: 사용자 질문
    
    Returns:
        str: Agent의 최종 응답
    """
    # 그래프 생성
    graph = create_agent_graph()
    
    # 초기 상태 설정
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    print(f"\n{'='*60}")
    print(f"🙋 사용자: {query}")
    print('='*60)
    
    # 그래프 실행
    # invoke(): 동기적으로 그래프를 실행하고 최종 상태를 반환
    result = graph.invoke(initial_state)
    
    # 최종 응답 추출
    final_message = result["messages"][-1]
    
    print(f"\n🤖 Agent: {final_message.content}")
    print('='*60)
    
    return final_message.content


def run_agent_with_stream(query: str):
    """
    스트리밍 모드로 Agent를 실행합니다.
    
    각 단계별로 중간 결과를 확인할 수 있습니다.
    
    Args:
        query: 사용자 질문
    """
    graph = create_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    print(f"\n{'='*60}")
    print(f"🙋 사용자: {query}")
    print('='*60)
    
    # stream(): 각 단계별 상태 변화를 순차적으로 반환
    # stream_mode="values"는 전체 상태를 반환
    # stream_mode="updates"는 변경된 부분만 반환
    for step, state in enumerate(graph.stream(initial_state, stream_mode="values")):
        print(f"\n📍 Step {step}")
        last_message = state["messages"][-1]
        print(f"   메시지 타입: {type(last_message).__name__}")
        
        if hasattr(last_message, "content") and last_message.content:
            print(f"   내용: {last_message.content[:100]}...")
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"   도구 호출: {[tc['name'] for tc in last_message.tool_calls]}")


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LangGraph 기본 Agent 예제")
    print("="*60)
    
    # 설정 확인 (제거됨: Local LLM 등 다양한 환경 지원을 위해 엄격한 키 검증 생략)
    # 필요한 경우 실행 시점에 오류로 포착
    pass

    # 연결 테스트 (제거됨: 실제 쿼리 실행 시 오류를 포착하여 처리)
    # from utils.llm_factory import get_llm, log_llm_error
    # test_llm = get_llm()
    
    # 테스트 쿼리 실행
    test_queries = [
        "서울의 날씨가 어때?",
        "123 * 456은 얼마야?",
        "안녕하세요! 반갑습니다.",
    ]
    
    from utils.llm_factory import log_llm_error
    
    for query in test_queries:
        try:
            run_agent(query)
        except Exception as e:
            # 오류 발생 시 상세 로깅
            # (여기서는 LLM 인스턴스를 직접 가져올 수 없으므로 None 전달하거나, 
            #  필요하다면 get_llm()을 호출하여 URL 정보를 가져될 수 있음)
            # 간단히 exception만 넘깁니다.
            log_llm_error(e)
            print(f"❌ 실행 중 오류가 발생했습니다. 로그를 확인하세요.")
        
        print("\n")
