# 1. 라이브러리 임포트 (필요한 도구들을 가져오는 단계)
import sys
from pathlib import Path
from typing import Annotated  # 변수 타입에 추가적인 정보(메타데이터)를 붙일 때 사용
from typing_extensions import TypedDict  # 데이터 구조의 형식을 지정하는 '사전(Dictionary)' 타입을 정의
from langgraph.graph import StateGraph, START, END  # 그래프를 생성하는 도구와 시작/종료 지점
from langgraph.graph.message import add_messages  # 새로운 메시지를 기존 리스트에 자동으로 합쳐주는 특수 함수
from langchain_openai import ChatOpenAI  # OpenAI 호환 LLM을 사용하기 위한 클래스
from langchain_core.tools import tool  # 파이썬 함수를 AI가 이해할 수 있는 '도구'로 변환해주는 도구
from langgraph.prebuilt import ToolNode, tools_condition  # 도구 실행 전용 노드와 조건부 실행 결정을 돕는 미리 만들어진 함수

# 프로젝트 루트를 path에 추가 (config 모듈 임포트를 위해)
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings  # 설정 로드 함수

# 2. State 정의 (프로그램이 기억해야 할 '메모리' 영역의 설계도)
# 목적: 챗봇이 대화하는 동안 어떤 정보를 들고 있을지 정의합니다.
class State(TypedDict):
    # messages라는 이름의 리스트를 관리합니다.
    # Annotated와 add_messages를 결합하면: "새 답변이 오면 기존 메시지 뒤에 누적해줘!"라는 뜻이 됩니다.
    messages: Annotated[list, add_messages]

# 3. 도구(Tool) 정의 (AI가 실제로 수행할 수 있는 구체적인 기술들)
@tool
def get_weather(location: str):
    """특정 지역의 현재 날씨와 기온 정보를 조회하는 도구입니다."""
    # (실제 API 연결 대신 예시 데이터를 반환하도록 설정됨)
    return f"{location}의 날씨는 맑음, 기온은 현재 25도입니다."

@tool
def search_web(query: str):
    """사용자가 궁금해하는 일반적인 지식이나 최신 정보를 웹에서 검색하는 도구입니다."""
    return f"'{query}' 검색 결과: LangGraph는 AI 에이전트를 조립하고 제어하는 강력한 프레임워크입니다."

# 정의한 도구들을 리스트로 묶어 관리합니다.
tools = [get_weather, search_web]

# 4. LLM(인공지능) 설정 및 기능 부여 (AI 두뇌 세팅)
# settings.py에서 base_url, api_key, model을 가져옵니다.
settings = get_settings()

# ChatOpenAI로 OpenAI 호환 LLM 생성 (로컬 LLM도 지원)
# .bind_tools(tools)를 통해 AI에게 "너는 이런 도구들도 쓸 수 있어!"라고 알려줍니다.
llm = ChatOpenAI(
    base_url=settings.openai_api_base,
    api_key=settings.openai_api_key,
    model=settings.openai_model
).bind_tools(tools)

# 5. 노드 함수 정의 (그래프의 각 칸에서 일어날 구체적인 행동)
def chatbot(state: State):
    """사용자의 질문을 읽고, LLM에게 전달하여 답변을 받아오는 함수입니다."""
    # 현재 상태(대화 내용)를 AI 모델에게 전달하여 실행합니다.
    response = llm.invoke(state["messages"])
    # AI의 응답을 다시 메시지 리스트 형식으로 반환합니다.
    return {"messages": [response]}

# 6. 그래프 구성 (워크플로우 지도 그리기)
# State 설계를 바탕으로 그래프 빌더 객체를 생성합니다.
graph_builder = StateGraph(State)

# (1) 노드 배치: "chatbot"이라는 작업 칸과 "tools"라는 작업 칸을 만듭니다.
graph_builder.add_node("chatbot", chatbot)  # 챗봇이 생각하는 단계
tool_node = ToolNode(tools=tools)  # AI가 '도구 써야지!'하면 실제로 실행해주는 단계
graph_builder.add_node("tools", tool_node)  # 실제 도구 실행 단계를 그래프에 추가

# (2) 시작점 설정: 프로그램이 시작되면 가장 먼저 "chatbot" 칸으로 갑니다.
graph_builder.add_edge(START, "chatbot")

# (3) 조건부 경로 설정 (중요): 
# 챗봇이 답변을 하고 나면 결과에 따라 길을 나눕니다.
# - AI가 '도구가 필요해'라고 하면 -> "tools" 칸으로 이동
# - AI가 '답변 끝'이라고 하면 -> 그래프 "종료(END)"
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # 이 함수가 다음에 어디로 갈지 자동으로 결정해줍니다.
)

# (4) 되돌아오는 경로 설정:
# 도구 실행("tools")이 끝나면, 그 결과를 가지고 다시 "chatbot"에게 가서 최종 답변을 완성하게 합니다.
graph_builder.add_edge("tools", "chatbot")

# 7. 실행 객체 생성 (지도 완공)
# 지금까지 그린 지도를 바탕으로 실제로 작동하는 '앱'을 컴파일합니다.
app = graph_builder.compile()

# --- 실행 참고 (사용 방법) ---
# 아래 코드를 주석 해제하면 실제로 테스트해볼 수 있습니다.
# if __name__ == "__main__":
#     from langchain_core.messages import HumanMessage
#     # 챗봇에게 날씨에 대해 물어보는 초기 상태를 만듭니다.
#     initial_state = {"messages": [HumanMessage(content="서울 날씨가 어때?")]}
#     # 그래프를 실행하며 각 단계별로 일어나는 일을 출력합니다.
#     for event in app.stream(initial_state):
#         print(event)
