# 🛠️ Utility: Vector Store Manager

이 모듈은 RAG 시스템에서 사용하는 Vector Store를 관리합니다. ChromaDB를 기본 백엔드로 사용하며, 문서 로드, 청킹(Chunking), 임베딩 및 유사도 검색 기능을 통합하여 제공합니다.

---

## 🔑 주요 기능

1. **지연 초기화 (Lazy Initialization)**: Vector Store가 실제로 필요한 시점에 인스턴스를 생성합니다.
2. **자동 텍스트 분할**: `RecursiveCharacterTextSplitter`를 사용하여 긴 문서를 최적의 크기로 자동 분할합니다.
3. **편리한 인터페이스**: 파일 로드, 텍스트 추가, 문서 추가, 검색 등을 단순화된 메서드로 제공합니다.
4. **Retriever 변환**: LangChain의 LCEL 체인에서 즉시 사용할 수 있는 `Retriever` 객체로 변환 가능합니다.

---

## 💻 사용 방법

### 1. 매니저 초기화
임베딩 모델은 `llm_factory`를 통해 자동으로 설정되거나 수동으로 주입할 수 있습니다.

```python
from utils.vector_store import VectorStoreManager

# 기본 설정으로 초기화 (메모리 저장)
manager = VectorStoreManager()

# 데이터 영구 저장을 위한 초기화
manager = VectorStoreManager(
    persist_directory="./chroma_db",
    collection_name="my_rag_docs"
)
```

### 2. 문서 추가 및 검색

```python
# 1. 텍스트 추가
manager.add_texts(["첫 번째 문서 내용", "두 번째 문서 내용"])

# 2. 파일에서 직접 로드
manager.load_from_file("my_document.txt")

# 3. 유사도 검색 (K=4)
results = manager.search("검색하고 싶은 내용")

for doc in results:
    print(f"찾은 내용: {doc.page_content}")
```

### 3. LangChain 체인과 통합

```python
# Retriever로 변환하여 체인에 연결
retriever = manager.as_retriever(search_kwargs={"k": 3})

# LCEL 구성 예시
# chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | output_parser
```

---

## 📐 핵심 메서드 정보

| 메서드 | 설명 |
|--------|------|
| `add_documents(docs)` | LangChain Document 객체 리스트를 분할하여 저장 |
| `split_text(text)` | 텍스트를 설정된 chunk_size에 맞춰 분할 |
| `search(query, k=4)` | 유사도가 높은 상위 k개의 문서를 반환 |
| `search_with_score(query)` | 검색 결과와 함께 유사도 점수(L2 distance) 반환 |
| `clear()` | 현재 컬렉션의 모든 데이터를 삭제하고 리셋 |

---

## ⚙️ 설정 파라미터

- `chunk_size`: 문서 분할 시 각 청크의 최대 길이 (기본값: 1000)
- `chunk_overlap`: 청크 간 겹치는 부분의 크기 (기본값: 200)
- `collection_name`: ChromaDB 내에서 구분할 컬렉션 이름

---

## 🔗 관련 모듈
- [LLM 팩토리](utils_llm_factory.md)
- [Naive RAG 예제](02_naive_rag.md)
