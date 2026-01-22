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
from typing import Annotated, Literal  # Annotated: 상태 업데이트 방식 지정, Literal: 값의 종류 제한

# 프로젝트 루트를 경로에 추가하여 내부 모듈(config, utils)을 불러올 수 있게 함
sys.path.insert(0, str(Path(__file__).parent.parent))

# 🔍 LangChain DEBUG 로깅 활성화 - LLM과 주고받는 raw 메시지 확인
import langchain
langchain.debug = True  # 상세 로그를 위해 다시 켭니다
# 또는 더 상세한 로그:
# import logging
# logging.getLogger("langchain").setLevel(logging.DEBUG)
# logging.getLogger("openai").setLevel(logging.DEBUG)
# logging.getLogger("httpx").setLevel(logging.DEBUG)

# LangChain: 대화 메시지 구조 및 도구 정의
from langchain_core.messages import HumanMessage, SystemMessage  # Human: 사용자 메시지, System: AI 지침
from langchain_core.tools import tool  # 파이썬 함수를 AI 도구로 변환하는 데코레이터

# LangGraph: 그래프 기반 에이전트 설계 및 실행
from langgraph.graph import StateGraph, MessagesState, START, END  # 그래프 빌더, 표준 상태, 시작/종료 지점
from langgraph.prebuilt import ToolNode, tools_condition  # 표준 도구 실행 노드 및 자동 라우팅 조건

# 프로젝트 유틸리티: 설정 로드 및 LLM 생성 팩토리
from config.settings import get_settings  # 중앙 설정(API 키, 모델명 등) 로드
from utils.llm_factory import get_llm, log_llm_error  # LLM 인스턴스 생성 및 오류 로깅 유틸리티
from utils.harmony_parser import parse_harmony_tool_call, clean_history_for_harmony  # GPT-OSS Harmony 유틸리티


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
    import json
    
    llm = get_llm()
    # 💡 vLLM/Local LLM 호환성: 병렬 도구 호출 비활성화 (많은 서버가 지원하지 않음)
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    
    # 시스템 메시지 정의
    sys_msg = SystemMessage(content="당신은 날씨 조회와 계산을 돕는 유용한 어시스턴트입니다.")

    # 메시지 리스트 구성
    messages = [sys_msg] + state["messages"]
    
    # 🧹 vLLM 호환성: LLM이 이해할 수 있는 클린한 포맷으로 변환 (History Cleaning)
    cleaned_messages = clean_history_for_harmony(messages)
    
    # LLM 호출
    response = llm_with_tools.invoke(cleaned_messages)
    
    # 🔍 디버깅 로그: LLM 응답 상세 분석
    print(f"\n{'='*60}")
    print(f"🔍 [DEBUG] LLM 응답 분석")
    print(f"{'='*60}")
    print(f"📌 response type: {type(response).__name__}")
    print(f"📌 response.content: {repr(response.content)}")
    print(f"📌 response.tool_calls: {response.tool_calls}")
    print(f"📌 response.additional_kwargs: {json.dumps(response.additional_kwargs, indent=2, ensure_ascii=False, default=str)}")
    
    # content가 JSON인지 확인
    if response.content and isinstance(response.content, str):
        try:
            parsed = json.loads(response.content)
            print(f"📌 content JSON 파싱 결과: {json.dumps(parsed, indent=2, ensure_ascii=False)}")
        except json.JSONDecodeError:
            print(f"📌 content는 JSON이 아님 (일반 텍스트)")
    print(f"{'='*60}\n")
    
    # 🔧 GPT-OSS Harmony 포맷 파싱: content의 JSON을 tool_calls로 변환
    response = parse_harmony_tool_call(response, tools)
    
    if response.tool_calls:
        print(f"🔧 [HARMONY] tool_calls 변환 완료: {[tc['name'] for tc in response.tool_calls]}")
    
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
        # invoke 모드로 실행 (스트리밍 대신)
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]}
        )
        
        final_msg = result["messages"][-1] if result.get("messages") else None
        
        if final_msg:
             print(f"\n🤖 최종 답변: {final_msg.content}")

    except Exception as e:
        log_llm_error(e)
        print("❌ 실행 중 오류가 발생했습니다.")


if __name__ == "__main__":
    print("\nLangGraph Basic Agent (Standard Pattern)")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    while True:
        try:
            query = input("🙋 질문을 입력하세요: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("quit", "exit", "q"):
                print("👋 Agent를 종료합니다.")
                break
            
            run_agent(query)
            
        except KeyboardInterrupt:
            print("\n👋 Agent를 종료합니다.")
            break
        except EOFError:
            print("\n👋 Agent를 종료합니다.")
            break
