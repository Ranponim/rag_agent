# 🔧 LangGraph API 레퍼런스

LangGraph에서 자주 사용되는 핵심 함수와 클래스에 대한 상세 가이드입니다.

---

## 📋 목차

- [Graph 구성](#graph-구성)
  - [StateGraph](#stategraph)
  - [MessagesState](#messagesstate)
- [Node 관리](#node-관리)
  - [add_node()](#add_node)
  - [ToolNode](#toolnode)
- [Edge 관리](#edge-관리)
  - [add_edge()](#add_edge)
  - [add_conditional_edges()](#add_conditional_edges)
- [실행](#실행)
  - [compile()](#compile)
  - [invoke()](#invoke)
  - [stream()](#stream)
- [상수](#상수)
  - [START와 END](#start와-end)
- [Memory](#memory)
  - [MemorySaver](#memorysaver)

---

## Graph 구성

### StateGraph

그래프를 구성하는 핵심 빌더 클래스입니다.

```python
from langgraph.graph import StateGraph

# 상태 정의
class MyState(TypedDict):
    messages: list
    data: str

# StateGraph 생성
graph = StateGraph(MyState)
```

**매개변수:**
| 매개변수 | 타입 | 설명 |
|----------|------|------|
| `state_schema` | TypedDict | 그래프 전체에서 공유할 상태의 스키마 |

**주요 메서드:**
- `add_node()` - 노드 추가
- `add_edge()` - 엣지 추가
- `add_conditional_edges()` - 조건부 엣지 추가
- `compile()` - 실행 가능한 그래프로 컴파일

---

### MessagesState

메시지 기반 상태를 위한 기본 타입입니다.

```python
from langgraph.graph import MessagesState

# MessagesState는 다음과 동일:
# class MessagesState(TypedDict):
#     messages: Annotated[list, add_messages]

graph = StateGraph(MessagesState)
```

**특징:**
- `messages` 필드가 자동으로 정의됨
- `add_messages` 리듀서로 메시지가 자동 누적됨
- 챗봇, Agent 구현 시 편리함

---

## Node 관리

### add_node()

그래프에 노드(작업 단위)를 추가합니다.

```python
# 함수를 노드로 추가
def my_node(state: MyState) -> dict:
    return {"data": "updated"}

graph.add_node("node_name", my_node)

# 함수명을 노드명으로 자동 사용
graph.add_node(my_node)  # 노드명: "my_node"
```

**시그니처:**
```python
add_node(
    node: str | Callable,       # 노드 이름 또는 함수
    action: Callable = None,    # 노드 이름 사용 시 실행할 함수
    metadata: dict = None,      # 메타데이터 (선택)
    retry_policy: RetryPolicy = None  # 재시도 정책 (선택)
)
```

**노드 함수 규칙:**
```python
def node_function(state: StateType) -> dict:
    """
    Args:
        state: 현재 그래프 상태 (전체 상태 딕셔너리)
    
    Returns:
        dict: 업데이트할 상태 필드만 포함
              (기존 상태와 병합됨)
    """
    # 상태에서 데이터 읽기
    current_value = state["field_name"]
    
    # 새 값 반환 (해당 필드만 업데이트됨)
    return {"field_name": new_value}
```

---

### ToolNode

도구 실행을 위한 특수 노드입니다.

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """도시의 날씨를 반환합니다."""
    return f"{city}: 맑음"

tools = [get_weather]
tool_node = ToolNode(tools)

graph.add_node("tools", tool_node)
```

**특징:**
- LLM의 `tool_calls`를 자동으로 처리
- 도구 실행 결과를 상태에 추가
- Agent 구현 시 필수 구성요소

---

## Edge 관리

### add_edge()

두 노드를 무조건적으로 연결합니다.

```python
from langgraph.graph import START, END

# 시작점에서 첫 노드로
graph.add_edge(START, "node_a")

# 노드 간 연결
graph.add_edge("node_a", "node_b")

# 마지막 노드에서 종료
graph.add_edge("node_b", END)
```

**시그니처:**
```python
add_edge(
    start_key: str | list[str],  # 시작 노드(들)
    end_key: str                 # 종료 노드
)
```

**여러 노드에서 하나로:**
```python
# node_a, node_b 모두 완료 후 node_c 실행
graph.add_edge(["node_a", "node_b"], "node_c")
```

---

### add_conditional_edges()

조건에 따라 다른 노드로 분기합니다.

```python
from typing import Literal

def router(state: MyState) -> Literal["path_a", "path_b", END]:
    """조건에 따라 다음 노드 결정"""
    if state["condition"]:
        return "path_a"
    elif state["other_condition"]:
        return "path_b"
    return END

graph.add_conditional_edges(
    "node_name",     # 시작 노드
    router,          # 라우터 함수
    # 경로 매핑 (선택, 반환값과 노드명이 같으면 생략 가능)
    {
        "path_a": "node_a",
        "path_b": "node_b",
        END: END
    }
)
```

**시그니처:**
```python
add_conditional_edges(
    source: str,              # 시작 노드
    path: Callable,           # 경로 결정 함수
    path_map: dict = None,    # 반환값 → 노드명 매핑 (선택)
)
```

**라우터 함수 패턴:**
```python
from typing import Literal

# 반환값 타입힌트로 가능한 경로 명시
def should_continue(state) -> Literal["continue", END]:
    if state["done"]:
        return END
    return "continue"
```

---

## 실행

### compile()

StateGraph를 실행 가능한 CompiledGraph로 변환합니다.

```python
# 기본 컴파일
compiled = graph.compile()

# 체크포인터 사용 (메모리 저장)
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
compiled = graph.compile(checkpointer=memory)
```

**시그니처:**
```python
compile(
    checkpointer: CheckpointSaver = None,  # 상태 저장소
    interrupt_before: list[str] = None,    # 실행 전 중단할 노드
    interrupt_after: list[str] = None,     # 실행 후 중단할 노드
)
```

**반환:** `CompiledStateGraph` (Runnable 인터페이스 구현)

---

### invoke()

그래프를 동기적으로 실행합니다.

```python
# 기본 실행
result = compiled.invoke({"question": "안녕?"})

# config 사용 (thread_id 등)
result = compiled.invoke(
    {"question": "안녕?"},
    config={"configurable": {"thread_id": "session-1"}}
)
```

**시그니처:**
```python
invoke(
    input: dict,              # 초기 상태
    config: RunnableConfig = None,  # 실행 설정
) -> dict                     # 최종 상태
```

---

### stream()

그래프를 스트리밍 모드로 실행합니다.

```python
# 기본 스트리밍 (노드별 업데이트)
for event in compiled.stream({"question": "안녕?"}):
    print(event)

# stream_mode 옵션
# "updates": 변경된 부분만 (기본값)
# "values": 전체 상태
for state in compiled.stream(input, stream_mode="values"):
    print(state["messages"][-1])
```

**시그니처:**
```python
stream(
    input: dict,
    config: RunnableConfig = None,
    stream_mode: str = "updates",  # "updates" | "values"
) -> Iterator
```

---

## 상수

### START와 END

그래프의 시작점과 종료점을 나타내는 특수 상수입니다.

```python
from langgraph.graph import START, END

# START: 그래프 진입점
graph.add_edge(START, "first_node")

# END: 그래프 종료점
graph.add_edge("last_node", END)

# 조건부 종료
def router(state) -> Literal["continue", END]:
    if state["done"]:
        return END
    return "continue"
```

---

## Memory

### MemorySaver

그래프 상태를 저장하고 복원하는 체크포인터입니다.

```python
from langgraph.checkpoint.memory import MemorySaver

# 메모리 기반 체크포인터 생성
memory = MemorySaver()

# 컴파일 시 체크포인터 추가
compiled = graph.compile(checkpointer=memory)

# thread_id로 대화 세션 구분
config = {"configurable": {"thread_id": "user-123"}}

# 첫 번째 메시지
result1 = compiled.invoke({"messages": [("user", "안녕")]}, config)

# 같은 thread_id로 이어서 대화 (이전 상태 유지)
result2 = compiled.invoke({"messages": [("user", "내 이름 뭐야?")]}, config)
```

**주요 기능:**
- 대화 히스토리 유지
- 상태 시점 복원 (time travel)
- 중단된 그래프 재개

---

## 그래프 패턴 예시

### 1. 단순 순차 실행

```python
graph.add_edge(START, "step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", END)
```

### 2. 조건부 분기

```python
graph.add_edge(START, "check")
graph.add_conditional_edges("check", router)
```

### 3. 루프 (자기 수정)

```python
graph.add_edge(START, "process")
graph.add_conditional_edges("process", should_retry)
graph.add_edge("retry", "process")  # 루프 백
```

### 4. 병렬 실행

```python
graph.add_edge(START, "branch_a")
graph.add_edge(START, "branch_b")
graph.add_edge(["branch_a", "branch_b"], "merge")
```

---

## 참고

- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
