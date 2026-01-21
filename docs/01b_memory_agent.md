# 📘 01b. Memory Agent - 대화 기록 유지

MemorySaver를 사용하여 대화 기록을 유지하고 세션을 관리하는 Agent입니다.

---

## 📋 학습 목표

1. MemorySaver 체크포인터 사용법
2. thread_id로 세션 분리
3. 대화 컨텍스트 유지
4. 이전 대화 참조

---

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
    graph = StateGraph(MessagesState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    
    # ⭐ 핵심: MemorySaver로 상태 저장 활성화
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    
    return compiled
```

### 세션별 대화
```python
def chat(graph, thread_id: str, message: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config  # thread_id 지정
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
