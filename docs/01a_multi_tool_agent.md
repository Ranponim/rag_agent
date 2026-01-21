# 📘 01a. Multi-Tool Agent - 다중 도구 Agent

여러 도구를 관리하고 LLM이 적절한 도구를 선택하는 Agent입니다.

---

## 📋 학습 목표

1. 다양한 종류의 도구 정의 방법
2. 도구 설명(docstring)의 중요성
3. 복잡한 질문에 대한 다중 도구 호출
4. 도구 실행 결과 처리

---

## 🔧 정의된 도구

| 도구 | 설명 |
|------|------|
| `get_weather` | 도시별 날씨 조회 |
| `calculate` | 수학 계산 |
| `search_knowledge` | 지식 베이스 검색 |
| `get_time` | 현재 시간 조회 |
| `translate` | 텍스트 번역 |

---

## 📐 핵심 코드

### 도구 정의
```python
@tool
def get_weather(city: str) -> str:
    """
    도시의 현재 날씨와 기온을 조회합니다.
    
    Args:
        city: 날씨를 조회할 도시명 (예: 서울, 부산, 제주)
    """
    # docstring이 LLM에게 도구 사용법을 알려줌
    ...
```

### Agent 노드
```python
def agent_node(state: MessagesState) -> dict:
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)  # 5개 도구 바인딩
    
    system_message = SystemMessage(content="""
    사용 가능한 도구:
    1. get_weather: 도시별 날씨 조회
    2. calculate: 수학 계산
    ...
    """)
    
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

---

## ✨ 핵심 포인트

1. **도구 설명(docstring)**: LLM이 도구를 선택할 때 참고
2. **복잡한 질문**: 여러 도구를 순차적으로 호출 가능
3. **도구 목록 관리**: 리스트로 도구를 관리하여 확장 용이

---

## 🔗 관련 문서

- [기본 Agent](01_basic_agent.md)
- [다음: Memory Agent](01b_memory_agent.md)
