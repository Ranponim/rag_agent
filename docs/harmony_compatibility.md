# 🔧 GPT-OSS (vLLM) Harmony 호환성 가이드

이 문서는 새로운 에이전트를 개발할 때 **GPT-OSS (vLLM 기반 로컬 모델)** 서버와의 호환성을 확보하는 방법에 대해 설명합니다. GPT-OSS는 표준 OpenAI API와 약간 다른 응답 포맷(Harmony)을 사용하므로, 에이전트 노드에서 적절한 전처리와 후처리가 필요합니다.

---

## ❓ 왜 호환성 파싱이 필요한가요?

1. **도구 호출 포맷의 차이**: 표준 OpenAI API는 도구 호출 정보를 `tool_calls`라는 전용 필드에 담아 보내지만, GPT-OSS(vLLM)는 답변 내용(`content`) 자체에 JSON 형식으로 담아 보내는 경우가 많습니다.
2. **메시지 역할 제한**: vLLM 서버는 `ToolMessage`나 특정 순서의 메시지 구성을 거부하고 400 Bad Request 에러를 낼 수 있습니다. 이를 방지하기 위해 메시지를 정제해야 합니다.

---

## 🛠️ 핵심 함수 소개

`utils/harmony_parser.py`에 정의된 두 가지 핵심 함수를 사용합니다.

### 1. `clean_history_for_harmony(messages)`
- **언제 사용하는가?**: LLM을 **호출하기 직전**에 사용합니다.
- **하는 일**: 
    - `ToolMessage`를 `HumanMessage`로 변환하여 서버 거부반응 방지
    - 빈 `content`를 가진 메시지에 기본 텍스트 삽입
    - `AIMessage`의 `tool_calls`를 `content`로 안전하게 이동

### 2. `parse_harmony_tool_call(response, tools)`
- **언제 사용하는가?**: LLM **호출 직후** 응답을 받았을 때 사용합니다.
- **하는 일**:
    - `content` 필드에 있는 JSON을 분석하여 랭그래프가 인식할 수 있는 `tool_calls` 객체로 복원
    - 이미 표준 `tool_calls`가 있다면 그대로 유지 (OpenAI와 로컬 LLM 모두 호환 가능하게 함)

---

## 📝 실제 적용 방법 (Step-by-Step)

새로운 에이전트 노드를 만들 때 아래와 같은 순서로 코드를 작성합니다.

### 1단계: 필요한 모듈 임포트
```python
from utils.harmony_parser import (
    clean_history_for_harmony, 
    parse_harmony_tool_call
)
```

### 2단계: 에이전트 노드 구현
```python
def my_agent_node(state: MessagesState):
    # 1. LLM 준비 및 도구 바인딩
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    
    # 2. 메시지 히스토리 준비
    messages = [SystemMessage(content="...")] + state["messages"]
    
    # ⭐ [단계 1: 전처리] vLLM 호환성을 위해 히스토리 정제
    cleaned_messages = clean_history_for_harmony(messages)
    
    # 3. LLM 호출
    response = llm_with_tools.invoke(cleaned_messages)
    
    # ⭐ [단계 2: 후처리] Harmony 포맷 응답을 표준 tool_calls로 변환
    response = parse_harmony_tool_call(response, tools)
    
    return {"messages": [response]}
```

---

## 💡 주의 사항

- **`parallel_tool_calls=False`**: 로컬 LLM 환경에서는 여러 도구를 한 번에 호출하는 기능이 불안정할 수 있으므로, `bind_tools` 호출 시 이 옵션을 `False`로 설정하는 것을 권장합니다.
- **원본 Content 유지**: `parse_harmony_tool_call`은 원본 텍스트를 파괴하지 않고 `tool_calls` 속성만 추가합니다. 이는 이후 대화 기록 유지에 필수적입니다.

---

## 🔗 관련 예제 파일
이 로직이 실제로 적용된 코드를 확인하려면 다음 파일을 참고하세요:
- [01_basic_agent.py](../examples/01_basic_agent.py)
- [01a_multi_tool_agent.py](../examples/01a_multi_tool_agent.py)
- [05_integrated_test.py](../examples/05_integrated_test.py)
