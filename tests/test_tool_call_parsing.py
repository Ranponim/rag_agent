# -*- coding: utf-8 -*-
"""
GPT-OSS Tool Call 파싱 테스트

이 스크립트는 GPT-OSS-20B의 tool call 응답 형식을 확인하고,
LangChain이 이를 어떻게 파싱하는지 테스트합니다.
"""

import sys
from pathlib import Path
import json

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from config.settings import get_settings


@tool
def get_weather(city: str) -> str:
    """특정 도시의 날씨 정보를 반환합니다."""
    return f"{city}은(는) 맑음, 15°C"


@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다."""
    return f"결과: {eval(expression)}"


def test_raw_response():
    """LLM의 raw 응답 형식을 확인합니다."""
    settings = get_settings()
    
    llm = ChatOpenAI(
        api_key=settings.openai_api_key or "dummy-key",
        model=settings.openai_model,
        base_url=settings.openai_api_base,
    )
    
    tools = [get_weather, calculate]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        SystemMessage(content="당신은 날씨 조회와 계산을 돕는 유용한 어시스턴트입니다."),
        HumanMessage(content="서울 날씨 어때?"),
    ]
    
    print("=" * 60)
    print("1. LLM 응답 분석 (tool binding 포함)")
    print("=" * 60)
    
    response = llm_with_tools.invoke(messages)
    
    print(f"\n📌 응답 타입: {type(response).__name__}")
    print(f"\n📌 response.content:")
    print(f"   Type: {type(response.content)}")
    print(f"   Value: {repr(response.content)}")
    
    print(f"\n📌 response.tool_calls:")
    if hasattr(response, 'tool_calls'):
        print(f"   Type: {type(response.tool_calls)}")
        print(f"   Value: {response.tool_calls}")
        print(f"   Length: {len(response.tool_calls) if response.tool_calls else 0}")
    else:
        print("   ❌ tool_calls 속성 없음")
    
    print(f"\n📌 response.additional_kwargs:")
    if hasattr(response, 'additional_kwargs'):
        print(f"   {json.dumps(response.additional_kwargs, indent=4, ensure_ascii=False)}")
    
    # content가 JSON 문자열인지 확인
    if response.content and isinstance(response.content, str):
        try:
            parsed = json.loads(response.content)
            print(f"\n📌 content를 JSON으로 파싱 가능:")
            print(f"   {json.dumps(parsed, indent=4, ensure_ascii=False)}")
        except json.JSONDecodeError:
            print(f"\n📌 content는 JSON이 아님")
    
    return response


def test_without_tools():
    """도구 없이 일반 응답을 테스트합니다."""
    settings = get_settings()
    
    llm = ChatOpenAI(
        api_key=settings.openai_api_key or "dummy-key",
        model=settings.openai_model,
        base_url=settings.openai_api_base,
    )
    
    messages = [
        SystemMessage(content="당신은 친절한 어시스턴트입니다."),
        HumanMessage(content="안녕하세요"),
    ]
    
    print("\n" + "=" * 60)
    print("2. 일반 대화 (도구 바인딩 없음)")
    print("=" * 60)
    
    response = llm.invoke(messages)
    
    print(f"\n📌 응답 타입: {type(response).__name__}")
    print(f"📌 content: {response.content}")
    
    return response


if __name__ == "__main__":
    print("\n🔬 GPT-OSS Tool Call 파싱 테스트\n")
    
    response1 = test_raw_response()
    response2 = test_without_tools()
    
    print("\n" + "=" * 60)
    print("3. 결론")
    print("=" * 60)
    
    if response1.tool_calls:
        print("\n✅ tool_calls가 정상적으로 파싱됨 - 문제없음")
    else:
        print("\n❌ tool_calls가 비어있음 - Harmony 포맷 파싱 필요")
        if response1.content:
            print(f"   content에 있는 값: {response1.content}")
