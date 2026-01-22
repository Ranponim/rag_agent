# 📘 04a. Adaptive RAG - 적응형 RAG

쿼리 복잡도를 분류하여 적절한 RAG 전략을 동적으로 선택하는 Adaptive RAG입니다.

---

## 📋 학습 목표

1. 쿼리 복잡도 분류 (단순/중간/복잡)
2. 전략별 다른 파이프라인 실행
3. 동적 라우팅
4. 비용-품질 트레이드오프

---

## 🖥️ CLI 실행 방법

이 예제는 **대화형 CLI 모드**로 실행됩니다.

```bash
python examples/04a_adaptive_rag.py
```

```
Adaptive RAG 예제 (CLI 모드)
질문의 난이도를 분석하여 가장 효율적인 방식으로 대답합니다.
종료하려면 'quit' 또는 'exit'를 입력하세요.

🙋 질문을 입력하세요: RAG와 파인튜닝의 차이는?
```

### 종료 방법
- `quit`, `exit`, 또는 `q` 입력
- `Ctrl+C` 키 입력

---

## 🔑 핵심 개념

### 복잡도별 전략

| 복잡도 | 예시 | 전략 |
|--------|------|------|
| **Simple** | "RAG가 뭐야?" | 검색 없이 직접 답변 |
| **Moderate** | "RAG의 장점은?" | 기본 RAG |
| **Complex** | "RAG와 Fine-tuning 비교 분석" | 다단계 RAG |

---

## 📐 그래프 구조

```mermaid
graph TD
    START --> classify[복잡도 분류]
    classify -->|simple| simple[직접 답변]
    classify -->|moderate| moderate[기본 RAG]
    classify -->|complex| complex[다단계 RAG]
    simple --> END
    moderate --> END
    complex --> END
```

---

## 📐 핵심 코드

### 복잡도 분류 (심사위원 AI)
```python
def classify_query_node(state: AdaptiveRAGState) -> dict:
    """[판별 단계] 질문을 읽고 '쉬움/보통/어려움' 중 하나로 분류합니다."""
    # AI 심사위원에게 질문의 난이도를 판단해달라고 지시합니다.
    # 1. "simple": 인사, 상식 질문
    # 2. "moderate": 일반 검색이 필요한 질문
    # 3. "complex": 다각도 분석이 필요한 복잡한 질문
    response = llm.invoke(...)
    
    return {"query_complexity": response.content.lower().strip()}
```

### 복잡 전략 (다단계 정밀 RAG)
```python
def complex_strategy_node(state: AdaptiveRAGState) -> dict:
    """[전략 3: 어려운 질문] 질문을 쪼개서 깊게 조사하고 분석 보고서를 만듭니다."""
    # 1. 어려운 질문을 해결하기 위한 2개의 세부 질문을 생성합니다.
    sub_queries = llm.invoke(...) 
    
    # 2. 메인 질문 + 세부 질문들로 지식 창고를 각각 검색합니다.
    for sq in sub_queries + [state["question"]]:
        docs = vs.search(query=sq, k=2)
        all_context.extend(docs)
    
    # 3. 모은 모든 정보를 합쳐서 심층 답변을 생성합니다.
    res = llm.invoke(...)
    return {"strategy_used": "Complex (다단계 정밀 RAG)", "answer": res.content}
```

---

## ✨ 핵심 포인트

1. **비용 효율**: 단순 질문에 RAG 불필요
2. **품질 최적화**: 복잡한 질문에 다단계 처리
3. **동적 라우팅**: LLM이 전략 결정

---

## 🔗 관련 문서

- [기본 Advanced RAG](04_advanced_rag.md)
