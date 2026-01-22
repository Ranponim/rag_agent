# -*- coding: utf-8 -*-
# 이 파일은 UTF-8 인코딩을 사용하여 한글이 깨지지 않도록 설정합니다. (초심자용 상세 주석 버전)

"""
============================================================================
📚 01c. Multi-Agent - 여러 Agent가 협업하는 시스템
============================================================================

이 예제는 여러 전문 Agent(Researcher, Analyst, Writer)가 팀을 이루어 
복잡한 미션을 함께 수행하는 '멀티 에이전트' 시스템을 구현합니다.

🎯 핵심 학습 포인트:
    1. 각자 맡은 역할(전문 분야)이 다른 여러 AI를 만드는 법.
    2. Supervisor(관리자): 팀장 AI가 업무 진행 상황을 보고 다음 담당자를 정하는 패턴.
    3. State(상태): 팀원들이 조사하고 분석한 결과를 한 장의 메모지에 계속 적어 공유하는 방식.
"""

# =============================================================================
# 📦 필수 라이브러리 임포트
# =============================================================================

import sys                              # 시스템 환경 제어용
from pathlib import Path                # 파일 경로 처리용
from typing import TypedDict, Literal, List  # 결과물 형식 정의용

# 프로젝트 최상위 폴더를 인식시켜 다른 폴더의 모듈을 불러오게 합니다.
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangChain 메시지 형식과 프롬프트 템플릿(지시서 양식)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# LangGraph의 핵심 순서도(그래프) 도구
from langgraph.graph import StateGraph, START, END

# 프로젝트 유틸리티
from config.settings import get_settings
from utils.llm_factory import get_llm, log_llm_error


# =============================================================================
# 📋 1. 공유 상태(State) 정의하기
# =============================================================================
# 팀 모든 멤버가 같이 쓰는 '공동 작업판'입니다.
# 한 명이 결과를 적으면 다른 멤버가 그걸 보고 이어서 일합니다.
# =============================================================================

class MultiAgentState(TypedDict):
    """팀 전체가 공유하는 메모장 양식입니다."""
    task: str                        # 처음 시킨 일 (주제)
    current_agent: str               # 지금 일하고 있거나 일해야 할 담당자 이름
    research_result: str             # 조사가 끝난 내용 (Researcher가 적음)
    analysis_result: str             # 분석이 끝난 내용 (Analyst가 적음)
    writing_result: str              # 최종 글쓰기 결과 (Writer가 적음)
    final_output: str                # 사용자에게 보여줄 마지막 답장
    agent_history: List[str]         # 누가 어떤 순서로 일했는지 기록 (기록용)


# =============================================================================
# 🤖 2. 전문 멤버(Agent) 노드 정의하기
# =============================================================================

def supervisor_node(state: MultiAgentState) -> dict:
    """
    [팀장] Supervisor: 팀원들의 진행 상황을 보고 다음 순서를 결정합니다.
    """
    print("\n🎯 [Supervisor] 업무 상황 체크 중... 다음엔 누구를 투입할까요?")
    
    llm = get_llm() # AI 모델 호출
    
    # 팀장에게 주는 지침 메모입니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 팀의 관리자(PM)입니다.
공동 작업판을 확인하고, 다음에 일할 사람을 한 단어로만 골라주세요.

[결정 규칙]
1. 기초 정보 조사가 안 되어 있다면 -> "researcher"
2. 정보 조사는 됐는데 전문 분석이 안 됐으면 -> "analyst"
3. 조사와 분석이 다 끝났는데 글 작성이 안 됐으면 -> "writer"
4. 최종 보고서까지 다 완성되었다면 -> "done"

현재 주제: {task}
진행 상황 요약:
- 연구 조사: {research_result}
- 데이터 분석: {analysis_result}
- 최종 작성: {writing_result}
"""),
    ])
    
    # AI팀장이 상황을 보고 다음 담당자 이름을 말합니다.
    response = (prompt | llm).invoke({
        "task": state["task"],
        "research_result": state.get("research_result") or "시작 전",
        "analysis_result": state.get("analysis_result") or "시작 전",
        "writing_result": state.get("writing_result") or "시작 전",
    })
    
    # AI가 말한 담당자 이름을 깔끔하게 정리합니다.
    next_agent = response.content.lower().strip().replace('"', '').replace('.', '')
    
    # 안전 장치: 단어에 오타가 있더라도 정확한 이름으로 맞춰줍니다.
    if "research" in next_agent: next_agent = "researcher"
    elif "analy" in next_agent: next_agent = "analyst"
    elif "write" in next_agent: next_agent = "writer"
    elif "done" in next_agent: next_agent = "done"
    
    print(f"   → 결정: 다음 업무는 '{next_agent}'님에게 맡깁니다.")
    
    # 결정된 담당자 이름과 일한 순서를 작업판에 업데이트합니다.
    return {
        "current_agent": next_agent,
        "agent_history": state.get("agent_history", []) + ["supervisor"],
    }


def researcher_node(state: MultiAgentState) -> dict:
    """
    [리서처] Researcher: 주제에 대한 팩트와 기초 정보를 수집합니다.
    """
    print("\n🔬 [Researcher] 관련 정보를 열심히 조사하고 있습니다...")
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 탐사 보도 전문 리서처입니다. 주제에 대해 구체적인 사실 관계를 풍부하게 조사하세요."),
        ("human", "주제: {task}"),
    ])
    
    # AI가 조사를 수행합니다.
    response = (prompt | llm).invoke({"task": state["task"]})
    
    # 조사한 내용을 'research_result' 칸에 적어 놓습니다.
    return {
        "research_result": response.content,
        "agent_history": state.get("agent_history", []) + ["researcher"],
    }


def analyst_node(state: MultiAgentState) -> dict:
    """
    [분석가] Analyst: 리서치된 내용을 바탕으로 인사이트(통찰)를 뽑아냅니다.
    """
    print("\n📊 [Analyst] 수집된 자료를 바탕으로 심층 분석을 시작합니다...")
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 냉철한 데이터 분석가입니다. 리서치 결과를 토대로 장점, 단점, 앞으로의 전망을 분석하세요."),
        ("human", "리서치 내용:\n{research_result}"),
    ])
    
    # 리서치 결과를 보고 분석합니다.
    response = (prompt | llm).invoke({"research_result": state["research_result"]})
    
    # 분석 결과를 'analysis_result' 칸에 적습니다.
    return {
        "analysis_result": response.content,
        "agent_history": state.get("agent_history", []) + ["analyst"],
    }


def writer_node(state: MultiAgentState) -> dict:
    """
    [작가] Writer: 조사와 분석 결과를 예쁜 보고서나 글 한 편으로 완성합니다.
    """
    print("\n✍️ [Writer] 모든 자료를 종합하여 최종 결과물을 작성하고 있습니다...")
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 전문 작가입니다. 리서치와 분석 데이터를 활용해 가독성 좋은 보고서나 깔끔한 요약본을 작성하세요."),
        ("human", "재료:\n- 조사 정보: {research_result}\n- 전문 분석: {analysis_result}"),
    ])
    
    # 모든 재료를 모아서 글을 씁니다.
    response = (prompt | llm).invoke({
        "research_result": state["research_result"],
        "analysis_result": state["analysis_result"]
    })
    
    # 최종 결과물을 'writing_result'와 'final_output'에 적습니다.
    return {
        "writing_result": response.content,
        "final_output": response.content,
        "agent_history": state.get("agent_history", []) + ["writer"],
    }


# =============================================================================
# 🔀 3. 길잡이(라우터) 함수
# =============================================================================

def route_by_supervisor(state: MultiAgentState) -> Literal["researcher", "analyst", "writer", "done"]:
    """팀장이 말한 다음 담당자 노드로 길을 안내해주는 신호등 역할입니다."""
    # 팀장이 current_agent 칸에 적어놓은 이름을 확인합니다.
    next_agent = state.get("current_agent", "done")
    
    # 그 이름이 목록에 있는 이름이면 그리로 보내고, 없으면 종료시킵니다.
    if next_agent in ["researcher", "analyst", "writer", "done"]:
        return next_agent
    
    return "done"


# =============================================================================
# 🔗 4. 협업 그래프 구성 (조직도 만들기)
# =============================================================================

def create_multi_agent_graph():
    """AI들이 서로 어떻게 일감을 주고받을지 화살표를 그립니다."""
    # 우리가 만든 양식(MultiAgentState)을 사용하는 흐름도를 준비합니다.
    builder = StateGraph(MultiAgentState)
    
    # 1. 팀원(노드)들을 배치합니다.
    builder.add_node("supervisor", supervisor_node) # 팀장
    builder.add_node("researcher", researcher_node) # 리서처
    builder.add_node("analyst", analyst_node)       # 분석가
    builder.add_node("writer", writer_node)         # 작가
    
    # 2. 시작하면 무조건 팀장(supervisor)에게 갑니다.
    builder.add_edge(START, "supervisor")
    
    # 3. 팀장이 상황을 보고 멤버를 호출합니다 (조건부 연결).
    builder.add_conditional_edges(
        "supervisor",          # 팀장 단계가 끝나면
        route_by_supervisor,   # 신호등(라우터)이 길을 묻습니다.
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "done": END        # "다 끝났다"고 하면 마침표(END)를 찍습니다.
        }
    )
    
    # 4. 업무를 마친 멤버는 다시 팀장에게 보고하러 돌아옵니다 (화살표).
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("writer", "supervisor")
    
    # 5. 이제 전체 협업 시스템을 조립합니다.
    return builder.compile()


# =============================================================================
# ▶️ 5. 실행 함수 (명령 내리기)
# =============================================================================

def run_team_task(task_query: str, team_graph):
    """지정한 업무를 AI 팀에게 시키고 그 결과를 구경합니다."""
    print(f"\n{'='*60}")
    print(f"📋 요청하신 업무: {task_query}")
    print(f"{'='*60}")
    
    # 처음 일을 시킬 때의 텅 빈 작업판 상태입니다.
    initial_state = {
        "task": task_query, # 주제만 적어 놓습니다.
        "current_agent": "",
        "research_result": "",
        "analysis_result": "",
        "writing_result": "",
        "final_output": "",
        "agent_history": []
    }
    
    try:
        # AI 팀 전체 시스템(그래프)을 가동합니다.
        result = team_graph.invoke(initial_state)
        
        # 일이 끝난 뒤의 최종 보고서를 출력합니다.
        print(f"\n{'━'*60}")
        print("🚩 업무 완료 보고")
        print(f"협업 순서: {' → '.join(result['agent_history'])}")
        print(f"{'━'*60}")
        
        print("\n📄 최종 결과물:")
        print("-" * 50)
        print(result["final_output"])
        print("-" * 50)
        
    except Exception as e:
        log_llm_error(e)
        print(f"❌ 협업 도중 문제가 생겼어요: {e}")


# =============================================================================
# 🚀 6. 메인 실행부 (CLI 인터페이스)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🤝 멀티 에이전트 협업 팀을 호출합니다! 🤝")
    print("리서처, 분석가, 작가가 힘을 합쳐 결과물을 만들어 드립니다.")
    print("- 'q'나 'exit'를 입력하면 팀이 해산합니다.\n")
    
    # 1. 협업 시스템을 한 번만 구성합니다.
    team_graph = create_multi_agent_graph()
    
    while True:
        try:
            # 일을 시킵니다.
            user_task = input("🙋 어떤 일을 시키시겠습니까?: ").strip()
            
            if not user_task: continue
                
            if user_task.lower() in ("quit", "exit", "q"):
                print("👋 안녕히 가세요! 다음에 또 일을 시켜주세요.")
                break
            
            # 작업을 시작합니다.
            run_team_task(user_task, team_graph)
            
        except KeyboardInterrupt:
            print("\n👋 급하게 종료합니다.")
            break
        except Exception as e:
            print(f"\n⚠️ 팀 내부 오류: {e}")
            break
