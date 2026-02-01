# 🛠️ Utility: Data Loader

RAG 시스템을 위한 데이터 로딩, 벡터화, 임베딩 통합 모듈입니다.
Vector Store 영속화와 파일 변경 자동 감지 기능을 제공합니다.

---

## 🔑 주요 기능

1. **다양한 파일 형식 지원**: TXT, MD, CSV, PDF, XLSX, JSON, JSONL
2. **Vector Store 영속화**: `./vector_db/` 폴더에 임베딩 데이터 자동 저장
3. **파일 변경 자동 감지**: `./rag` 폴더의 파일 추가/수정 시 자동 재임베딩
4. **Collection 기반 관리**: 같은 collection_name으로 임베딩 데이터 재사용

---

## 💻 사용 방법

### 1. 기본 사용법 (원스톱)

```python
from utils.data_loader import get_rag_vector_store

# ./rag 폴더의 모든 파일을 자동으로 임베딩
vs = get_rag_vector_store(collection_name="my_rag")

# 검색
results = vs.search("LangGraph란?", k=3)
```

### 2. 파라미터 옵션

```python
vs = get_rag_vector_store(
    collection_name="my_rag",      # 컬렉션 이름 (필수)
    source_dir="./rag",            # 소스 폴더 (기본값: ./rag)
    persist_dir="./vector_db",     # 저장 폴더 (기본값: ./vector_db)
    embedding_provider=None,       # 임베딩 제공자 (None=환경변수)
    force_reload=False             # 강제 재임베딩 여부
)
```

### 3. 강제 재임베딩

파일 변경 없이도 다시 임베딩하려면:

```python
vs = get_rag_vector_store(collection_name="my_rag", force_reload=True)
```

---

## 📂 지원 파일 형식

| 확장자 | 로더 | 처리 방식 |
|--------|------|----------|
| `.txt` | TextLoader | 전체 텍스트를 하나의 문서로 |
| `.md` | TextLoader | 전체 텍스트를 하나의 문서로 |
| `.csv` | CSVLoader | 각 행을 별도 문서로 |
| `.pdf` | PyPDFLoader | 페이지별 문서 분할 |
| `.xlsx` | UnstructuredExcelLoader | 엑셀 내용 추출 |
| `.json` | JSONLoader | JSON 전체를 하나의 문서로 |
| `.jsonl` | JSONLineLoader | **한 줄씩 별도 문서로** (커스텀) |

> **참고**: JSONL 파일은 한 줄씩 개별 임베딩되어 세밀한 검색이 가능합니다.

---

## 🔄 동작 원리

### 자동 변경 감지

```
1. ./rag 폴더의 파일 해시 계산
2. 이전 해시와 비교
3. 변경 시 → 재임베딩
4. 동일 시 → 기존 임베딩 재사용
```

### 저장 구조

```
vector_db/
├── naive_rag/           ← collection_name="naive_rag"
│   ├── chroma.sqlite3   ← ChromaDB 데이터
│   └── .folder_hash     ← 파일 변경 감지용 해시
├── entity_rag/          ← collection_name="entity_rag"
└── integrated_rag/      ← collection_name="integrated_rag"
```

---

## 🧩 RAGDataLoader 클래스

직접 제어가 필요한 경우 클래스를 사용할 수 있습니다.

```python
from utils.data_loader import RAGDataLoader

# 로더 초기화
loader = RAGDataLoader(source_dir="./rag", encoding="utf-8")

# 모든 파일 로드
documents = loader.load_all()

# 지원 확장자 확인
extensions = loader.get_supported_extensions()
```

---

## 📐 JSONLineLoader (커스텀)

JSONL 파일을 한 줄씩 처리하는 커스텀 로더입니다.

```python
from utils.data_loader import JSONLineLoader

loader = JSONLineLoader(file_path="data.jsonl", encoding="utf-8")
docs = loader.load()
# 100줄 → 100개의 Document
```

---

## ⚙️ 환경 변수

| 변수 | 설명 |
|------|------|
| `EMBEDDING_PROVIDER` | 임베딩 제공자 (openai/ollama) |
| `OLLAMA_EMBEDDING_MODEL` | Ollama 임베딩 모델명 |

---

## 🔗 관련 모듈

- [Vector Store Manager](utils_vector_store.md) - 벡터 DB 관리
- [LLM Factory](utils_llm_factory.md) - 임베딩 모델 생성
- [Naive RAG](02_naive_rag.md) - 기본 RAG 예제
