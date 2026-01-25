# -*- coding: utf-8 -*-
# 이 파일은 UTF-8 인코딩을 사용하여 한글이 깨지지 않도록 설정합니다. (초심자용 상세 주석 버전)

"""
============================================================================
📚 01a. Multi-Tool Agent - 다중 도구를 활용하는 Agent
============================================================================

이 예제는 여러 개의 도구를 관리하고, LLM이 적절한 도구를 선택하도록 하는
Agent를 구현합니다.

🎯 핵심 학습 포인트:
    1. 다양한 도구를 어떻게 정의하고 AI에게 알려주는지 배웁니다.
    2. AI가 도구를 선택할 때 참고하는 설명(docstring)의 중요성을 이해합니다.
    3. AI가 스스로 도구를 쓰고 결과를 받아 다시 대답하는 흐름을 익힙니다.
"""

# =============================================================================
# 📦 필수 라이브러리 임포트 (필요한 도구 꾸러미 가져오기)
# =============================================================================

import sys                              # 파이썬 시스템 환경을 제어하는 모듈입니다.
from pathlib import Path                # 컴퓨터의 파일 경로를 다루기 쉽게 해주는 모듈입니다.
from typing import Literal              # 특정 텍스트 값만 허용하도록 타입을 정의할 때 씁니다.

# 현재 실행 중인 파일의 부모 디렉토리(루트 폴더)를 파이썬 경로에 추가합니다.
# 이렇게 해야 다른 폴더에 있는 config나 utils 모듈을 불러올 수 있습니다.
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangChain에서 대화 메시지(사람, 시스템 메시지) 형식을 가져옵니다.
from langchain_core.messages import HumanMessage, SystemMessage
# 파이썬 함수를 AI가 쓸 수 있는 '도구'로 변환해주는 장식자(@tool)를 가져옵니다.
from langchain_core.tools import tool

# LangGraph의 핵심 구성 요소들을 가져옵니다.
from langgraph.graph import StateGraph, MessagesState, START, END
# 도구를 실행하는 노드와, 도구 사용 여부를 결정하는 조건 함수를 가져옵니다.
from langgraph.prebuilt import ToolNode, tools_condition

# 프로젝트 내부 설정과 LLM 생성 도우미를 가져옵니다.
from config.settings import get_settings
from utils.llm_factory import get_llm, log_llm_error


# =============================================================================
# 🛠️ 1. 다양한 도구(Tool) 정의하기
# =============================================================================
# AI에게 "이런 함수가 있으니 필요할 때 써라"라고 알려주는 과정입니다.
# 함수 아래의 큰따옴표 안의 설명(docstring)이 AI가 이 도구를 고르는 기준이 됩니다.
# =============================================================================

@tool # 이 함수를 AI가 쓸 수 있는 도구로 등록합니다.
def get_weather(city: str) -> str:
    """
    특정 도시의 현재 날씨와 기온을 조회합니다.
    (AI는 이 설명을 읽고 날씨 질문이 들어왔을 때 이 함수를 사용합니다.)
    
    Args:
        city: 날씨를 조회할 도시의 이름 (예: 서울, 부산, 제주)
    """
    # 실제 API 대신 미리 준비한 연습용 데이터입니다.
    weather_data = {
        "서울": "맑음, 15°C, 습도 45%",
        "부산": "흐림, 18°C, 습도 60%",
        "제주": "비, 20°C, 습도 80%",
    }
    # 입력받은 도시의 날씨를 찾아서 반환하고, 없으면 오류 메시지를 보냅니다.
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")


@tool # 수학 계산을 전문으로 하는 도구입니다.
def calculate(expression: str) -> str:
    """
    수학 표현식(예: 2+3*4)을 계산합니다. 사칙연산과 제곱 등을 지원합니다.
    """
    try:
        # 보안을 위해 숫자와 사칙연산 기호 등 안전한 문자만 골라냅니다.
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "오류: 허락되지 않는 문자가 포함되어 있습니다."
            
        # 문자열로 된 수식을 계산하여 결과를 뽑아냅니다.
        result = eval(expression)
        return f"계산 결과: {result}"
    except Exception as e:
        # 계산 중 틀리거나 오류가 나면 그 내용을 알려줍니다.
        return f"계산 오류: {str(e)}"


@tool # 내부 정보를 찾아보는 검색 도구입니다.
def search_knowledge(query: str) -> str:
    """
    AI, LangGraph, RAG 등 기술적인 개념에 대해 내부 지식을 검색합니다.
    """
    # 우리가 미리 입력해둔 지식 사전입니다.
    knowledge_base = {
        "langgraph": "LangGraph는 AI 에이전트의 흐름을 그래프로 설계하게 해주는 라이브러리입니다.",
        "rag": "RAG는 외부 문서를 실시간으로 찾아 답변을 보강하는 기술입니다.",
        "agent": "에이전트는 목표를 위해 스스로 도구를 쓰고 판단하는 시스템입니다.",
    }
    
    # 사용자가 물어본 키워드가 사전에 있는지 확인합니다.
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value # 찾으면 설명을 돌려줍니다.
    
    # 못 찾으면 아래 메시지를 보냅니다.
    return f"'{query}'에 대한 정보가 지식 베이스에 없습니다."


@tool # 현재 시간을 알려주는 도구입니다.
def get_time(timezone: str = "KST") -> str:
    """현재 날짜와 시간을 확인합니다."""
    from datetime import datetime # 시간 관련 모듈을 함수 안에서 불러옵니다.
    now = datetime.now() # 지금 이 순간의 시각을 가져옵니다.
    # 보기 좋게 연-월-일 시:분:초 형식으로 바꿔서 알려줍니다.
    return f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool # 간단한 번역을 도와주는 도구입니다.
def translate(text: str, target_lang: str = "en") -> str:
    """텍스트를 영어(en)나 일본어(ja)로 번역합니다."""
    # 간단한 번역 사전입니다.
    translations = {
        "안녕하세요": {"en": "Hello", "ja": "こんにちは"},
        "감사합니다": {"en": "Thank you", "ja": "ありがとうございます"},
        "잘 가": {"en": "Goodbye", "ja": "さようなら"},
    }
    
    # 입력받은 문자가 우리 사전에 있는지 확인합니다.
    text_lower = text.strip().lower()
    if text_lower in translations and target_lang in translations[text_lower]:
        return f"번역: {translations[text_lower][target_lang]}"
    
    # 사전에 없으면 연동이 안 되어 있다고 정중히 알립니다.
    return f"'{text}' 번역 서비스가 아직 준비되지 않았습니다."


# 🔗 만들어진 도구들을 하나로 묶어 리스트로 만듭니다.
# 나중에 AI에게 "자, 네가 쓸 수 있는 도구 목록이야"라고 전달할 때 사용합니다.
tools = [get_weather, calculate, search_knowledge, get_time, translate]


# =============================================================================
# 🤖 2. Agent 노드 정의 (AI의 뇌 역할)
# =============================================================================
# 이 함수는 AI가 질문을 분석하고, 도구를 쓸지 그냥 대답할지 정하는 단계입니다.
# =============================================================================

def agent_node(state: MessagesState) -> dict:
    """질문을 받고 무엇을 할지 결정하는 '생각' 노드입니다."""
    # 1. 사용할 AI 모델을 가져옵니다.
    llm = get_llm()
    # 2. AI에게 우리가 만든 도구 목록(tools)을 연결해줍니다.
    # 한번에 여러 도구를 부르지 않도록(parallel_tool_calls=False) 설정합니다.
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    
    # 3. AI의 정체성(페르소나)을 설정하는 기본 지침을 만듭니다.
    system_message = SystemMessage(content="""당신은 다재다능한 비서입니다.
- 필요한 도구를 적극적으로 활용해서 질문에 답하세요.
- 계산이 필요하면 계산기를, 모르는 정보는 검색을, 날씨는 날씨 도구를 쓰세요.
- 친절하게 한글로 대답해 주세요.
""")
    
    # 4. [기본 지침] + [이전까지 나눈 대화들]을 합쳐서 AI에게 전달할 메시지 통을 만듭니다.
    messages = [system_message] + state["messages"]
    
    # 5. AI에게 메시지를 보내고 답변을 받습니다.
    response = llm_with_tools.invoke(messages)
    
    # 만약 AI가 도구를 쓰기로 했다면, 무엇을 하려는지 콘솔(검은 창)에 보여줍니다.
    if response.tool_calls:
        print(f"🔧 [Agent] 생각 중... { [tc['name'] for tc in response.tool_calls] } 도구 사용!")
    
    # 업데이트된 메시지를 담아 다음 단계로 넘깁니다.
    return {"messages": [response]}


# =============================================================================
# 🔀 3. 워크플로우 그래프 구성 (흐름도 그리기)
# =============================================================================
# "시작하면 AI에게 가고, 도구가 필요하면 도구 노드로 가라"는 길을 만듭니다.
# =============================================================================

def create_multi_tool_agent():
    """에이전트의 작동 순서도를 만들고 컴파일(실행 준비)합니다."""
    # 흐름도를 그릴 수 있는 캔버스(StateGraph)를 준비합니다. 대화 상태를 공유합니다.
    builder = StateGraph(MessagesState)
    
    # 1. 노드(단계)를 추가합니다. 'agent'는 생각하는 단계, 'tools'는 도구를 쓰는 단계입니다.
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools)) # 미리 정의된 tools를 실행해주는 노드입니다.
    
    # 2. 시작점(START)에서 'agent' 단계로 화살표를 긋습니다.
    builder.add_edge(START, "agent")
    
    # 3. 조건에 따라 길을 나눕니다 (라우팅).
    builder.add_conditional_edges(
        "agent",          # agent 단계가 끝나면
        tools_condition,  # AI의 대답에 도구 사용 요청이 있는지 확인해서
        {
            "tools": "tools", # 도구가 필요하면 'tools' 노드로 가고,
            END: END          # 필요 없으면 바로 대화를 끝냅니다(END).
        }
    )
    
    # 4. 도구를 쓴 다음에는 다시 결과를 가지고 'agent'에게 돌아가서 답변을 완성하게 합니다.
    builder.add_edge("tools", "agent")
    
    # 5. 완성된 흐름도를 묶어서 실행 가능한 상태로 만듭니다.
    return builder.compile()


# =============================================================================
# ▶️ 4. 답변 실행 및 화면 출력 함수
# =============================================================================

def run_agent_interactive(query: str, graph):
    """사용자 질문을 입력받아 AI의 최종 답변을 보여줍니다."""
    print(f"\n{'='*60}")
    print(f"🙋 질문: {query}")
    print(f"{'='*60}")
    
    try:
        # 우리가 만든 흐름도(graph)를 실행(invoke)합니다.
        # 사람의 질문(HumanMessage)을 담아서 시작합니다.
        result = graph.invoke({"messages": [HumanMessage(content=query)]})
        
        # 전체 대화 기록 중 가장 마지막 메시지가 AI의 최종 답변입니다.
        final_answer = result["messages"][-1].content
        
        # 화면에 예쁘게 출력합니다.
        print(f"\n🤖 답변: {final_answer}")
        
    except Exception as e:
        # 도중에 오류가 나면 기록으로 남기고 사용자에게 알립니다.
        log_llm_error(e)
        print(f"❌ 오류가 났어요: {str(e)}")


# =============================================================================
# 🚀 5. 프로그램 실제 시작 부분
# =============================================================================

if __name__ == "__main__":
    # 안내 메시지를 출력합니다.
    print("\n" + "🌟 다중 도구 에이전트를 시작합니다! 🌟")
    print("날씨, 시간, 계산기, 검색, 번역 중 무엇이든 물어보세요.")
    print("끝내고 싶으면 'q' 또는 'exit'라고 입력하세요.\n")
    
    # 1. 전체 흐름도를 한 번 미리 만들어 둡니다.
    agent_graph = create_multi_tool_agent()
    
    # 2. 무한 반복하며 질문을 받습니다.
    while True:
        try:
            # 키보드로 질문을 입력받습니다.
            user_input = input("🙋 물어볼 것을 적어주세요: ").strip()
            
            # 아무것도 안 적고 엔터 치면 다시 물어봅니다.
            if not user_input: continue
                
            # '그만' 혹은 'q'라고 하면 프로그램을 종료합니다.
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 안녕히 가세요! 다음에 또 만나요.")
                break
                
            # 입력받은 질문으로 에이전트를 실행합니다.
            run_agent_interactive(user_input, agent_graph)
            
        except KeyboardInterrupt:
            # Ctrl+C를 눌렀을 때의 처리입니다.
            print("\n👋 강제로 프로그램을 종료합니다.")
            break
        except Exception as e:
            # 예기치 못한 에러가 나면 종료합니다.
            print(f"\n⚠️ 큰 오류가 발생해 종료합니다: {e}")
            break
