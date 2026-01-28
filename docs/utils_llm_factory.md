# 🛠️ Utility: LLM Factory

이 모듈은 다양한 LLM 제공자(OpenAI, Ollama 등)를 위한 팩토리 패턴을 구현합니다. 프로젝트 전체에서 일관된 방식으로 LLM과 임베딩 모델을 생성하고 관리할 수 있게 도와줍니다.

---

## 🔑 주요 기능

1. **멀티 제공자 지원**: OpenAI API뿐만 아니라 Ollama 등을 통한 로컬 임베딩 모델도 지원합니다.
2. **싱글톤 캐싱**: 동일한 설정의 LLM이나 임베딩 인스턴스를 중복 생성하지 않고 캐싱하여 효율적으로 재사용합니다.
3. **환경 변수 기반 설정**: `.env` 파일의 설정을 자동으로 읽어와 복잡한 설정 없이 인스턴스를 반환합니다.
4. **상세한 오류 로깅**: 연결 실패, API 오류 등 발생 시 사용자 친화적인 안내 메시지를 출력합니다.

---

## 💻 사용 방법

### 1. LLM 인스턴스 가져오기
가장 기본적인 방법으로, 설정된 기본 LLM(주로 Local LLM 또는 GPT-4o-mini)을 가져옵니다.

```python
from utils.llm_factory import get_llm

# 기본 설정된 LLM 가져오기
llm = get_llm()

# 특정 파라미터로 가져오기
llm = get_llm(temperature=0.7)
```

### 2. 임베딩 모델 가져오기
환경 변수(`EMBEDDING_PROVIDER`) 설정에 따라 OpenAI 또는 Ollama 임베딩을 가져옵니다.

```python
from utils.llm_factory import get_embeddings

# 설정에 따른 임베딩 모델 가져오기
embeddings = get_embeddings()

# 특정 제공자 강제 지정
ollama_embeddings = get_embeddings(provider="ollama")
```

---

## ⚙️ 주요 환경 변수 (.env)

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `OPENAI_API_BASE` | LLM 서버 주소 (Local LLM 사용 시 필수) | `http://localhost:1234/v1` |
| `OPENAI_MODEL` | 사용할 LLM 모델명 | `local-model` |
| `EMBEDDING_PROVIDER` | 임베딩 제공자 (`openai` 또는 `ollama`) | `openai` |
| `OLLAMA_EMBEDDING_MODEL` | Ollama에서 사용할 임베딩 모델명 | `nomic-embed-text` |
| `OLLAMA_EMBEDDING_BASE_URL` | Ollama 서버 주소 | `http://localhost:11434` |

---

## 📐 클래스 구조: LLMFactory

`LLMFactory`는 내부적으로 실제 인스턴스 생성을 담당하는 클래스입니다.

- `create_openai_llm()`: OpenAI 호환 인터페이스를 가진 LLM 생성
- `create_openai_embeddings()`: OpenAI 호환 임베딩 생성
- `create_ollama_embeddings()`: Ollama 전용 임베딩 생성

---

## 🔗 관련 모듈
- [Vector Store 매니저](utils_vector_store.md)
- [기본 개념](../README.md)
