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

import os

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# LangGraph 프리빌트 에이전트 생성 도구
from langgraph.prebuilt import create_react_agent

# LangChain 컴포넌트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

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
    # 모델 초기화 (도구 바인딩은 create_react_agent가 내부적으로 처리함)
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    # 에이전트의 역할과 지침 설정 (페르소나 정의)
    system_prompt = "당신은 날씨 정보와 간단한 계산을 도와주는 유용한 비서입니다. 모든 답변은 한국어로 친절하게 하세요."
    
    # create_react_agent를 사용하여 한 줄로 그래프 생성
    # 참고: LangGraph 1.0+에서는 'prompt' 파라미터를 사용하여 시스템 프롬프트를 설정합니다.
    # (이전 버전의 'state_modifier'는 deprecated 되었습니다.)
    app = create_react_agent(
        model, 
        tools=tools, 
        prompt=system_prompt
    )
    
    return app

# 3. 실행부 (Execution) - CLI 대화형 인터페이스
# 사용자 요청에 따라 로컬에서 직접 실행하실 수 있도록 구성하였습니다.
if __name__ == "__main__":
    # 에이전트 생성
    app = create_agent()
    
    print("=" * 50)
    print("🤖 LangGraph ReAct 에이전트 (CLI 대화 모드)")
    print("=" * 50)
    print("날씨 정보와 계산을 도와드립니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    # CLI 대화 루프
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("👤 You: ").strip()
            
            # 종료 조건 확인
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 대화를 종료합니다. 안녕히 가세요!")
                break
            
            # 빈 입력 처리
            if not user_input:
                print("⚠️  메시지를 입력해주세요.\n")
                continue
            
            # 에이전트 호출
            inputs = {"messages": [HumanMessage(content=user_input)]}
            result = app.invoke(inputs)
            
            # 응답 출력
            if "messages" in result:
                print("\n🤖 Agent: ", end="")
                # content만 추출하여 깔끔하게 출력
                print(result["messages"][-1].content)
            print()  # 줄바꿈
            
        except KeyboardInterrupt:
            # Ctrl+C로 종료 시
            print("\n\n👋 대화를 종료합니다. 안녕히 가세요!")
            break
        except Exception as e:
            print(f"\n❌ [오류 발생] {e}")
            print("팁: 로컬 LLM 서버(LM Studio 등)의 연결 상태를 확인해주세요.\n")
