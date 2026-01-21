# -*- coding: utf-8 -*-
"""
01c. Multi-Agent - 여러 Agent가 협업하는 시스템

이 예제는 여러 전문 Agent가 협력하여 복잡한 작업을 수행하는
Multi-Agent 시스템을 구현합니다.

학습 목표:
    1. 여러 Agent를 노드로 구성
    2. Supervisor 패턴으로 작업 분배
    3. Agent 간 상태 공유
    4. 복잡한 워크플로우 설계

실행: python examples/01c_multi_agent.py
"""

import sys
from pathlib import Path
from typing import TypedDict, Literal, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from config.settings import get_settings
from utils.llm_factory import get_llm


# =============================================================================
# 1. 상태 정의
# =============================================================================

class MultiAgentState(TypedDict):
    """멀티 에이전트 시스템의 공유 상태"""
    task: str                        # 원본 작업
    current_agent: str               # 현재 활성 에이전트
    research_result: str             # 리서치 결과
    analysis_result: str             # 분석 결과
    writing_result: str              # 작성 결과
    final_output: str                # 최종 출력
    agent_history: List[str]         # 에이전트 실행 히스토리


# =============================================================================
# 2. 전문 Agent 노드들
# =============================================================================

def supervisor_node(state: MultiAgentState) -> dict:
    """
    Supervisor Agent: 작업을 분석하고 적절한 Agent에게 할당
    
    역할:
    - 작업 요구사항 분석
    - 다음에 실행할 Agent 결정
    - 전체 워크플로우 조율
    """
    print("\n🎯 [Supervisor] 작업 분석 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 팀을 이끄는 Supervisor입니다.
        
작업을 분석하고 다음 중 하나를 선택하세요:
- "researcher": 정보 수집이 필요한 경우
- "analyst": 데이터 분석이 필요한 경우
- "writer": 결과물 작성이 필요한 경우
- "done": 모든 작업이 완료된 경우

현재 상태:
- 작업: {task}
- 리서치 결과: {research_result}
- 분석 결과: {analysis_result}
- 작성 결과: {writing_result}

다음에 실행할 에이전트를 "researcher", "analyst", "writer", "done" 중 하나만 답하세요."""),
        ("human", "어떤 에이전트가 다음에 작업해야 할까요?"),
    ])
    
    response = (prompt | llm).invoke({
        "task": state["task"],
        "research_result": state.get("research_result", "없음"),
        "analysis_result": state.get("analysis_result", "없음"),
        "writing_result": state.get("writing_result", "없음"),
    })
    
    # 응답에서 에이전트 이름 추출
    content = response.content.lower()
    
    if "writer" in content and state.get("analysis_result"):
        next_agent = "writer"
    elif "analyst" in content and state.get("research_result"):
        next_agent = "analyst"
    elif "researcher" in content and not state.get("research_result"):
        next_agent = "researcher"
    elif state.get("writing_result"):
        next_agent = "done"
    elif not state.get("research_result"):
        next_agent = "researcher"
    elif not state.get("analysis_result"):
        next_agent = "analyst"
    elif not state.get("writing_result"):
        next_agent = "writer"
    else:
        next_agent = "done"
    
    print(f"   → 다음 Agent: {next_agent}")
    
    return {
        "current_agent": next_agent,
        "agent_history": state.get("agent_history", []) + ["supervisor"],
    }


def researcher_node(state: MultiAgentState) -> dict:
    """
    Researcher Agent: 정보 수집 전문
    
    역할:
    - 주제에 대한 정보 조사
    - 관련 데이터 수집
    - 핵심 사실 정리
    """
    print("\n🔬 [Researcher] 정보 수집 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 전문 리서처입니다.
주어진 주제에 대해 핵심 정보를 조사하고 정리하세요.

조사 결과를 다음 형식으로 정리하세요:
1. 주요 개념
2. 핵심 사실 (3-5개)
3. 관련 키워드"""),
        ("human", "다음 주제를 조사해주세요: {task}"),
    ])
    
    response = (prompt | llm).invoke({"task": state["task"]})
    
    result = response.content
    print(f"   → 리서치 완료: {result[:100]}...")
    
    return {
        "research_result": result,
        "agent_history": state.get("agent_history", []) + ["researcher"],
    }


def analyst_node(state: MultiAgentState) -> dict:
    """
    Analyst Agent: 데이터 분석 전문
    
    역할:
    - 수집된 정보 분석
    - 패턴 및 인사이트 도출
    - 결론 도출
    """
    print("\n📊 [Analyst] 데이터 분석 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 전문 분석가입니다.
리서처가 수집한 정보를 분석하고 인사이트를 도출하세요.

분석 결과를 다음 형식으로 정리하세요:
1. 핵심 인사이트 (2-3개)
2. 장단점 분석
3. 결론"""),
        ("human", """원본 작업: {task}

리서치 결과:
{research_result}

위 내용을 분석해주세요."""),
    ])
    
    response = (prompt | llm).invoke({
        "task": state["task"],
        "research_result": state["research_result"],
    })
    
    result = response.content
    print(f"   → 분석 완료: {result[:100]}...")
    
    return {
        "analysis_result": result,
        "agent_history": state.get("agent_history", []) + ["analyst"],
    }


def writer_node(state: MultiAgentState) -> dict:
    """
    Writer Agent: 콘텐츠 작성 전문
    
    역할:
    - 분석 결과를 이해하기 쉽게 작성
    - 최종 결과물 생성
    - 포맷팅 및 정리
    """
    print("\n✍️ [Writer] 결과물 작성 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 전문 작가입니다.
리서치와 분석 결과를 바탕으로 명확하고 이해하기 쉬운 문서를 작성하세요.

다음 형식으로 작성하세요:
## 요약
(1-2문장 요약)

## 주요 내용
(핵심 포인트 정리)

## 결론
(최종 결론)"""),
        ("human", """원본 작업: {task}

리서치 결과:
{research_result}

분석 결과:
{analysis_result}

위 내용을 바탕으로 최종 문서를 작성해주세요."""),
    ])
    
    response = (prompt | llm).invoke({
        "task": state["task"],
        "research_result": state["research_result"],
        "analysis_result": state["analysis_result"],
    })
    
    result = response.content
    print(f"   → 작성 완료: {result[:100]}...")
    
    return {
        "writing_result": result,
        "final_output": result,
        "agent_history": state.get("agent_history", []) + ["writer"],
    }


# =============================================================================
# 3. 라우터 함수
# =============================================================================

def route_by_supervisor(state: MultiAgentState) -> Literal["researcher", "analyst", "writer", "done"]:
    """Supervisor가 결정한 다음 Agent로 라우팅"""
    next_agent = state.get("current_agent", "researcher")
    
    if next_agent == "done":
        return "done"
    
    return next_agent


# =============================================================================
# 4. 그래프 구성
# =============================================================================

def create_multi_agent_graph():
    """
    Multi-Agent 그래프 생성
    
    구조:
        START → supervisor → (researcher | analyst | writer | done) → supervisor → ...
        
        supervisor가 작업을 분배하고, 각 전문 Agent가 처리 후 
        다시 supervisor로 돌아가서 다음 Agent를 결정
    """
    graph = StateGraph(MultiAgentState)
    
    # 노드 추가
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    
    # 시작 → supervisor
    graph.add_edge(START, "supervisor")
    
    # supervisor → 조건부 분기
    graph.add_conditional_edges(
        "supervisor",
        route_by_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst", 
            "writer": "writer",
            "done": END,
        }
    )
    
    # 각 전문 Agent → supervisor (루프)
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")
    
    print("✅ Multi-Agent 그래프 컴파일 완료!")
    return graph.compile()


# =============================================================================
# 5. 실행
# =============================================================================

def run_multi_agent(task: str) -> str:
    """Multi-Agent 시스템 실행"""
    graph = create_multi_agent_graph()
    
    initial_state = {
        "task": task,
        "current_agent": "",
        "research_result": "",
        "analysis_result": "",
        "writing_result": "",
        "final_output": "",
        "agent_history": [],
    }
    
    print(f"\n{'='*60}")
    print(f"📋 작업: {task}")
    print('='*60)
    
    result = graph.invoke(initial_state)
    
    print(f"\n{'='*60}")
    print("📌 Agent 실행 순서:")
    print(f"   {' → '.join(result['agent_history'])}")
    print('='*60)
    print("\n📄 최종 결과:")
    print(result["final_output"])
    print('='*60)
    
    return result["final_output"]


if __name__ == "__main__":
    from utils.llm_factory import log_llm_error
    
    print("\n" + "="*60)
    print("Multi-Agent 예제 - 협업 시스템")
    print("="*60)
    
    try:
        # 복잡한 작업 실행
        task = "LangGraph의 장단점을 분석하고, 언제 사용해야 하는지 보고서를 작성해주세요."
        run_multi_agent(task)
        
    except Exception as e:
        log_llm_error(e)
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
