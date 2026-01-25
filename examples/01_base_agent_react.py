# -*- coding: utf-8 -*-
"""
LangGraph 최신 ReAct 구조 베이스 코드 (Modern create_react_agent)

이 예제는 LangGraph에서 제공하는 `create_react_agent` 프리빌트(prebuilt) 함수를 활용하여
가장 빠르고 간결하게 에이전트를 구축하는 현대적인 방식을 보여줍니다.

학습 목표:
1. create_react_agent를 이용한 복잡한 그래프 구성 자동화
2. 시스템 프롬프트(System Prompt) 설정 방법
3. 간단한 도구(Tool) 결합 및 실행
"""

import sys
from pathlib import Path

# LangGraph 프리빌트 에이전트 생성 도구
from langgraph.prebuilt import create_react_agent

# LangChain 컴포넌트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# 프로젝트 설정 로드 (API Key, Base URL 등)
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import get_settings

# 1. 도구 정의 (Tool Definition)
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

# 그래프에서 사용할 도구들을 리스트로 묶어줍니다.
tools = [get_weather, calculate]

# 2. 에이전트 생성 (Agent Setup)
def create_agent():
    settings = get_settings()
    
    # 모델 초기화 (도구 바인딩은 create_react_agent가 내부적으로 처리함)
    model = ChatOpenAI(
        base_url=settings.openai_api_base,
        api_key=settings.openai_api_key,
        model=settings.openai_model
    )
    
    # 에이전트의 역할과 지침 설정 (페르소나 정의)
    system_prompt = "당신은 날씨 정보와 간단한 계산을 도와주는 유용한 비서입니다. 모든 답변은 한국어로 친절하게 하세요."
    
    # create_react_agent를 사용하여 한 줄로 그래프 생성
    # 참고: LangGraph 1.0+에서는 'prompt' 파라미터를 사용하여 시스템 프롬프트를 설정합니다.
    # (이전 버전의 'state_modifier'는 deprecated 되었습니다.)
    agent_executor = create_react_agent(
        model, 
        tools=tools, 
        prompt=system_prompt
    )
    
    return agent_executor

# 3. 실행부 (Execution)
# 사용자 요청에 따라 로컬에서 직접 실행하실 수 있도록 구성하였습니다.
if __name__ == "__main__":
    # 에이전트 생성
    agent = create_agent()
    
    # 테스트 질문 생성 (날씨와 계산 요청)
    inputs = {"messages": [HumanMessage(content="서울 날씨를 알려주고, 25 * 4 결과도 계산해줘.")]}
    
    print("--- 에이전트 실행 시작 (ReAct 방식) ---")
    # invoke 실행 (최종 결과만 확인 - 환경 문제로 인해 stream 대신 사용)
    try:
        # invoke는 그래프의 모든 단계를 거친 후 최종 상태를 반환합니다.
        result = agent.invoke(inputs)
        
        if "messages" in result:
            # 최종 응답(AI의 답변) 메시지만 출력
            result["messages"][-1].pretty_print()
            
    except Exception as e:
        print(f"\n[오류 발생] 실행 중 문제가 발생했습니다: {e}")
        print("팁: LM Studio나 Ollama 등 로컬 LLM 서버가 활성화되어 있는지 확인해주세요.")
    
    print("--- 에이전트 실행 종료 ---")
