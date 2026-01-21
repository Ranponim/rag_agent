# -*- coding: utf-8 -*-
"""
01a. Multi-Tool Agent - 다중 도구를 활용하는 Agent

이 예제는 여러 개의 도구를 관리하고, LLM이 적절한 도구를 선택하도록 하는
Agent를 구현합니다.

학습 목표:
    1. 다양한 종류의 도구 정의 방법
    2. 도구 설명(docstring)의 중요성
    3. 복잡한 질문에 대한 다중 도구 호출
    4. 도구 실행 결과 처리

실행: python examples/01a_multi_tool_agent.py
"""

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from config.settings import get_settings
from utils.llm_factory import get_llm


# =============================================================================
# 1. 다양한 도구 정의
# =============================================================================

@tool
def get_weather(city: str) -> str:
    """
    도시의 현재 날씨와 기온을 조회합니다.
    
    Args:
        city: 날씨를 조회할 도시명 (예: 서울, 부산, 제주)
    
    Returns:
        str: 해당 도시의 날씨 정보
    """
    weather_data = {
        "서울": "맑음, 15°C, 습도 45%",
        "부산": "흐림, 18°C, 습도 60%",
        "제주": "비, 20°C, 습도 80%",
        "인천": "맑음, 14°C, 습도 50%",
        "대전": "구름 조금, 16°C, 습도 55%",
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")


@tool
def calculate(expression: str) -> str:
    """
    수학 표현식을 계산합니다. 사칙연산, 거듭제곱 등을 지원합니다.
    
    Args:
        expression: 계산할 수학식 (예: "2 + 3 * 4", "10 ** 2")
    
    Returns:
        str: 계산 결과
    """
    try:
        # 안전한 연산만 허용
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "오류: 허용되지 않는 문자가 포함되어 있습니다."
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def search_knowledge(query: str) -> str:
    """
    내부 지식 베이스에서 정보를 검색합니다. 일반적인 개념이나 정의를 찾을 때 사용합니다.
    
    Args:
        query: 검색할 키워드나 질문
    
    Returns:
        str: 검색 결과
    """
    knowledge_base = {
        "langgraph": "LangGraph는 상태 기반 Multi-Actor 애플리케이션을 구축하기 위한 프레임워크입니다.",
        "rag": "RAG(Retrieval-Augmented Generation)는 검색된 문서로 LLM 응답을 보강하는 기법입니다.",
        "agent": "AI Agent는 LLM을 사용해 스스로 판단하고 도구를 선택하여 작업을 수행하는 시스템입니다.",
        "embedding": "임베딩은 텍스트를 벡터로 변환하여 의미적 유사도를 계산할 수 있게 합니다.",
        "vector store": "Vector Store는 임베딩 벡터를 저장하고 유사도 검색을 수행하는 데이터베이스입니다.",
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    
    return f"'{query}'에 대한 정보를 찾을 수 없습니다."


@tool
def get_time(timezone: str = "KST") -> str:
    """
    현재 시간을 반환합니다.
    
    Args:
        timezone: 시간대 (기본값: KST)
    
    Returns:
        str: 현재 시간
    """
    from datetime import datetime
    now = datetime.now()
    return f"현재 시간 ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def translate(text: str, target_lang: str = "en") -> str:
    """
    간단한 번역을 수행합니다. (데모용 - 실제로는 번역 API 사용)
    
    Args:
        text: 번역할 텍스트
        target_lang: 목표 언어 코드 (en, ko, ja)
    
    Returns:
        str: 번역된 텍스트
    """
    # 데모용 간단한 번역
    translations = {
        "안녕하세요": {"en": "Hello", "ja": "こんにちは"},
        "감사합니다": {"en": "Thank you", "ja": "ありがとうございます"},
        "hello": {"ko": "안녕하세요", "ja": "こんにちは"},
    }
    
    text_lower = text.lower()
    if text_lower in translations and target_lang in translations[text_lower]:
        return f"번역 결과: {translations[text_lower][target_lang]}"
    
    return f"'{text}'를 {target_lang}로 번역: [번역 결과 - 실제 API 연동 필요]"


# 모든 도구를 리스트로 관리
tools = [get_weather, calculate, search_knowledge, get_time, translate]


# =============================================================================
# 2. Agent 노드 정의
# =============================================================================

def agent_node(state: MessagesState) -> dict:
    """
    Agent 노드: 다양한 도구를 사용할 수 있는 LLM 호출
    
    여러 도구가 바인딩되어 있으며, LLM이 질문에 따라 
    적절한 도구를 선택합니다.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    system_message = SystemMessage(content="""당신은 다양한 도구를 활용할 수 있는 만능 AI 어시스턴트입니다.

사용 가능한 도구:
1. get_weather: 도시별 날씨 조회
2. calculate: 수학 계산
3. search_knowledge: 지식 검색
4. get_time: 현재 시간 조회
5. translate: 텍스트 번역

사용자의 질문을 분석하여 적절한 도구를 선택하고 활용하세요.
복잡한 질문은 여러 도구를 순차적으로 사용할 수 있습니다.
""")
    
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


# =============================================================================
# 3. 라우터 함수
# =============================================================================

def should_continue(state: MessagesState) -> Literal["tools", END]:
    """도구 호출 여부에 따라 다음 노드 결정"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        print(f"🔧 도구 호출: {tool_names}")
        return "tools"
    
    print("✅ 최종 응답 완료")
    return END


# =============================================================================
# 4. 그래프 구성
# =============================================================================

def create_multi_tool_agent():
    """Multi-Tool Agent 그래프 생성"""
    graph = StateGraph(MessagesState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    
    print("✅ Multi-Tool Agent 컴파일 완료!")
    return graph.compile()


# =============================================================================
# 5. 실행
# =============================================================================

def run_agent(query: str) -> str:
    """Agent 실행"""
    graph = create_multi_tool_agent()
    
    print(f"\n{'='*60}")
    print(f"🙋 질문: {query}")
    print('='*60)
    
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    final_response = result["messages"][-1].content
    
    print(f"\n🤖 응답: {final_response}")
    print('='*60)
    
    return final_response


if __name__ == "__main__":
    from utils.llm_factory import log_llm_error
    
    print("\n" + "="*60)
    print("Multi-Tool Agent 예제")
    print("="*60)
    
    test_queries = [
        "서울의 날씨는 어때?",
        "123 * 456 + 789는 얼마야?",
        "RAG가 뭐야?",
        "지금 몇 시야?",
        "서울 날씨 알려주고, 15도에서 화씨로 변환하면 얼마야? (공식: F = C * 9/5 + 32)",
    ]
    
    for query in test_queries:
        try:
            run_agent(query)
        except Exception as e:
            log_llm_error(e)
            print(f"❌ 오류: {e}")
        print()
