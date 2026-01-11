# 📘 02. Naive RAG - 기본 RAG 파이프라인

가장 기본적인 RAG(Retrieval-Augmented Generation) 파이프라인 구현입니다.

---

## 📋 목차

- [개요](#개요)
- [RAG 아키텍처](#rag-아키텍처)
- [핵심 구성요소](#핵심-구성요소)
- [코드 분석](#코드-분석)
- [한계점과 개선 방향](#한계점과-개선-방향)
- [연습 문제](#연습-문제)

---

## 개요

### RAG란?

**RAG (Retrieval-Augmented Generation)**는:

1. 사용자 질문과 관련된 **문서를 검색**하고
2. 검색된 문서를 **컨텍스트로 제공**하여
3. LLM이 **더 정확한 답변을 생성**하도록 하는 기법

### 왜 RAG가 필요한가?

| 문제 | RAG 해결책 |
|------|-----------|
| LLM의 지식은 학습 시점까지만 | 최신 문서 검색으로 보완 |
| 도메인 특화 지식 부족 | 전문 문서 DB 활용 |
| 환각(Hallucination) 문제 | 검색된 문서에 기반한 답변 |

---

## RAG 아키텍처

### 그래프 구조

```mermaid
graph LR
    START((START)) --> retrieve[검색<br/>Retrieve]
    retrieve --> generate[생성<br/>Generate]
    generate --> END((END))
```

### 데이터 흐름

```
질문 → Vector Store 검색 → 관련 문서 → 프롬프트 구성 → LLM → 답변
```

---

## 핵심 구성요소

### 1. Vector Store

문서를 벡터로 저장하고 유사도 검색을 수행합니다.

```python
from utils.vector_store import VectorStoreManager

manager = VectorStoreManager(
    embeddings=embeddings,
    collection_name="my_rag",
    chunk_size=500,      # 청크 크기
    chunk_overlap=100,   # 청크 간 중복
)

# 문서 추가
manager.add_texts(["문서1", "문서2"])

# 검색
docs = manager.search("질문", k=3)
```

### 2. 임베딩 (Embedding)

텍스트를 벡터로 변환하는 모델입니다.

```python
from utils.llm_factory import get_embeddings

embeddings = get_embeddings()

# 텍스트 → 벡터
vector = embeddings.embed_query("안녕하세요")
# 결과: [0.012, -0.034, 0.056, ...]
```

### 3. 청킹 (Chunking)

긴 문서를 적절한 크기로 분할합니다.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)

chunks = splitter.split_text(long_document)
```

**청킹이 중요한 이유:**
- LLM 컨텍스트 길이 제한
- 더 정확한 유사도 검색
- 관련 부분만 효율적으로 전달

---

## 코드 분석

### State 정의

```python
class RAGState(TypedDict):
    question: str           # 사용자 질문
    context: str            # 검색된 컨텍스트
    documents: List[Document]  # 검색된 문서 객체
    answer: str             # 최종 답변
```

### Retrieve 노드

```python
def retrieve_node(state: RAGState) -> dict:
    """문서 검색"""
    manager = get_vector_store()
    
    # 유사도 검색
    documents = manager.search(
        query=state["question"],
        k=3  # 상위 3개
    )
    
    # 컨텍스트 문자열 생성
    context = "\n\n".join([
        f"[문서 {i}]\n{doc.page_content}"
        for i, doc in enumerate(documents, 1)
    ])
    
    return {"documents": documents, "context": context}
```

### Generate 노드

```python
def generate_node(state: RAGState) -> dict:
    """답변 생성"""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """컨텍스트 기반으로 답변하세요.
        
컨텍스트:
{context}"""),
        ("human", "{question}"),
    ])
    
    response = (prompt | llm).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    
    return {"answer": response.content}
```

---

## 한계점과 개선 방향

### Naive RAG의 한계

| 한계 | 설명 |
|------|------|
| 검색 품질 미검증 | 관련 없는 문서도 그대로 사용 |
| 단일 검색 시도 | 재검색 불가 |
| 답변 검증 없음 | 환각 가능성 |
| 쿼리 그대로 사용 | 쿼리 최적화 없음 |

### 개선 방향

```mermaid
graph TD
    subgraph Naive RAG
        A[검색] --> B[생성]
    end
    
    subgraph Advanced RAG
        C[쿼리 변환] --> D[검색]
        D --> E[관련성 평가]
        E -->|관련 없음| F[재검색/웹검색]
        E -->|관련 있음| G[생성]
        G --> H[환각 검사]
        H -->|환각| F
        H -->|정상| I[완료]
    end
```

---

## 실행 결과

### 테스트 쿼리

```
🙋 질문: LangGraph란 무엇인가요?
============================================================

🔍 검색 중: 'LangGraph란 무엇인가요?'
   → 3개 문서 발견

💭 답변 생성 중...
   → 답변 생성 완료

📚 검색된 문서 수: 3

🤖 답변:
LangGraph는 LangChain 팀에서 개발한 라이브러리로, 상태를 가진 
다중 행위자 애플리케이션을 구축하기 위한 도구입니다. 
주요 특징으로는 상태 관리, 사이클 지원, 세밀한 제어 등이 있습니다.
```

---

## 연습 문제

### 1. 다양한 문서 형식

PDF, CSV 등 다른 형식의 문서를 로드해보세요.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()
```

### 2. 검색 결과 개수 변경

`k` 값을 조절해 검색 결과 수가 답변에 미치는 영향을 확인하세요.

### 3. 청크 크기 실험

`chunk_size`와 `chunk_overlap`을 조절해 최적 값을 찾아보세요.

---

## 다음 단계

➡️ [03. Entity RAG](03_entity_rag.md) - 엔티티 기반 하이브리드 검색
