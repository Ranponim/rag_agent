# 📘 01b. Memory Agent - 대화 기록 유지

MemorySaver를 사용하여 대화 기록을 유지하고 세션을 관리하는 Agent입니다.

---

## 📋 학습 목표

1. MemorySaver 체크포인터 사용법
2. thread_id로 세션 분리
3. 대화 컨텍스트 유지
4. 이전 대화 참조

---

## 🖥️ CLI 실행 방법

이 예제는 **대화형 CLI 모드**로 실행됩니다.

```bash
python examples/01b_memory_agent.py
```

```
Memory Agent 예제 (CLI 모드)
이 Agent는 당신과 나눈 대화를 기억합니다.
종료하려면 'quit' 또는 'exit'를 입력하세요.

🙋 [Thread: user_session_01] 질문: 안녕, 내 이름은 철수야.
🙋 [Thread: user_session_01] 질문: 내 이름이 뭐야?
```

### 특수 명령어
- `/thread [세션ID]`: 대화 세션을 변경합니다. (예: `/thread room_02`)

### 종료 방법
- `quit`, `exit`, 또는 `q` 입력
- `Ctrl+C` 키 입력

---

> [!IMPORTANT]
> **GPT-OSS (vLLM) 호환성**: 로컬 LLM 서버를 사용하는 경우 [Harmony 호환성 가이드](harmony_compatibility.md)를 참고하여 응답 파싱 및 메시지 정제를 적용하세요.

## 🔑 핵심 개념

### MemorySaver
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
compiled = graph.compile(checkpointer=memory)
```

### thread_id
```python
# 같은 thread_id → 대화 연속
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [msg]}, config=config)
```

---

## 📐 핵심 코드

### 그래프 컴파일 (메모리 활성화)
```python
def create_memory_agent():
    """메모리 기능이 장착된 에이전트 순서도를 만듭니다."""
    builder = StateGraph(MessagesState)
    
    # ... 노드 및 엣지 추가 ...
    
    # ⭐ 핵심: 대화 저장소(MemorySaver) 만들기
    # 이 객체가 프로그램이 켜져 있는 동안 대화 내용을 기억해줍니다.
    memory = MemorySaver()
    
    # 그래프를 완성(컴파일)할 때 이 저장소를 'checkpointer'로 전달합니다.
    return builder.compile(checkpointer=memory)
```

### 세션별 대화 (thread_id 활용)
```python
def run_chat(graph, thread_id: str, query: str):
    """지정한 대화방 ID(thread_id)를 사용하여 대화를 나눕니다."""
    # 같은 thread_id를 지정하면 LangGraph가 해당 ID의 이전 상태를 자동으로 로드합니다.
    config = {"configurable": {"thread_id": thread_id}}
    
    # invoke(입력, 설정)을 통해 이전 기억을 불러와 대화를 진행합니다.
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    return result["messages"][-1].content
```

---

## 🧪 사용 예시

```python
# User A와 대화 (thread-A)
chat(graph, "thread-A", "내 이름은 철수야")
chat(graph, "thread-A", "내 이름이 뭐야?")  # → "철수" 기억

# User B와 대화 (별도 세션 thread-B)
chat(graph, "thread-B", "내 이름은 영희야")
chat(graph, "thread-B", "내 이름이 뭐야?")  # → "영희" 기억 (thread-A와 분리)
```

---

## ✨ 핵심 포인트

1. **checkpointer 지정**: `compile(checkpointer=memory)`
2. **thread_id로 세션 분리**: 다른 사용자/대화를 분리
3. **상태 복원**: 같은 thread_id로 호출하면 이전 상태 복원

---

## 🔗 관련 문서

- [이전: Multi-Tool Agent](01a_multi_tool_agent.md)
- [다음: Multi-Agent](01c_multi_agent.md)
