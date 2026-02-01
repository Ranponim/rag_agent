# -*- coding: utf-8 -*-
"""
============================================================================
📚 02. Naive RAG 예제 - 기본 RAG 파이프라인 구현
============================================================================

LangGraph를 사용하여 가장 기본적인 검색-생성(Retrieve-Generate) 파이프라인을 구축합니다.
StateGraph를 활용하여 검색 결과와 생성된 답변을 상태로 관리하는 방법을 학습합니다.

🎯 학습 목표:
    1. RAG의 표준 파이프라인 (Retrieve → Generate) 구현
    2. 사용자 정의 State(TypedDict) 설계
    3. Vector Store 연동 및 검색 노드 구현

💡 RAG란?
    Retrieval-Augmented Generation (검색 증강 생성)
    - LLM에게 질문만 던지는 것이 아니라
    - 먼저 관련 문서를 검색(Retrieve)하고
    - 그 문서를 참고하여 답변을 생성(Generate)하는 기법
    - 이를 통해 LLM이 학습하지 않은 최신 정보도 답변 가능

그래프 구조:
    START → retrieve → generate → END
    (선형 구조, 분기 없음)

실행 방법:
    python examples/02_naive_rag.py
    
    실행 후 CLI에서 질문을 입력하면 RAG Agent가 응답합니다.
    종료: 'quit', 'exit', 또는 'q' 입력
"""

# =============================================================================
# 📦 필수 라이브러리 임포트
# =============================================================================

# Python 표준 라이브러리
import sys                              # 시스템 경로 조작용
from pathlib import Path                # 파일 경로를 객체지향적으로 다루는 라이브러리
from typing import TypedDict, List, Annotated  
# - TypedDict: 딕셔너리의 키와 값 타입을 정의하는 타입 힌트
# - List: 리스트 타입 힌트 (예: List[str]는 문자열 리스트)
# - Annotated: 타입에 추가 메타데이터 부여

# 프로젝트 루트를 Python 경로에 추가하여 내부 모듈 import 가능하게 함
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 파일에서 환경변수 로드
import os
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# 🔗 LangChain 핵심 모듈 임포트
# -----------------------------------------------------------------------------

from langchain_openai import ChatOpenAI # LLM 모델 클래스
from langchain_core.documents import Document
# Document: 검색된 텍스트를 담는 표준 객체
# - page_content: 실제 텍스트 내용
# - metadata: 출처, 페이지 번호 등 부가 정보

from langchain_core.prompts import ChatPromptTemplate
# ChatPromptTemplate: LLM에게 전달할 프롬프트의 템플릿
# 변수 자리를 {변수명}으로 지정하고 나중에 값을 채움

# -----------------------------------------------------------------------------
# 🔗 LangGraph 핵심 모듈 임포트
# -----------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END
# - StateGraph: 상태 기반 그래프 빌더
# - START: 그래프 시작점
# - END: 그래프 종료점

# -----------------------------------------------------------------------------
# 🔗 프로젝트 내부 유틸리티 임포트
# -----------------------------------------------------------------------------

from utils.llm_factory import get_embeddings, log_llm_error
# - get_embeddings: 텍스트를 벡터로 변환하는 임베딩 모델
# - log_llm_error: LLM 오류 상세 로깅

from utils.vector_store import VectorStoreManager
# Vector Store(벡터 DB) 관리 클래스
# 텍스트를 벡터로 변환하여 저장하고, 유사한 문서를 검색


# =============================================================================
# 📋 1. State 정의
# =============================================================================
#
# RAG 파이프라인에서 노드들 사이에 전달되는 데이터 구조입니다.
# TypedDict를 사용하면 딕셔너리에 어떤 키가 있고, 각 키의 값은 어떤 타입인지 명시할 수 있습니다.
#
# 💡 MessagesState vs TypedDict:
#    - MessagesState: 대화 메시지 리스트를 관리 (Agent 예제에서 사용)
#    - TypedDict: 커스텀 필드가 필요할 때 사용 (RAG에서는 documents, answer 등)
# =============================================================================

class RAGState(TypedDict):
    """
    RAG 파이프라인의 상태 정의
    
    각 노드는 이 상태를 읽고, 필요한 필드를 업데이트합니다.
    
    필드 설명:
    - question: 사용자가 입력한 질문 (입력)
    - documents: 검색된 관련 문서 리스트 (중간 결과)
    - answer: LLM이 생성한 최종 답변 (출력)
    """
    question: str                    # 사용자 질문 (Input)
    documents: List[Document]        # 검색된 문서들 (Intermediate)
    answer: str                      # 최종 답변 (Output)


# =============================================================================
# 🗄️ 2. Vector Store 초기화 (공통 모듈 사용)
# =============================================================================
#
# 💡 데이터 로딩 및 벡터화는 utils/data_loader.py 공통 모듈에서 처리합니다.
# - 다양한 파일 형식 지원 (TXT, MD, CSV, PDF, XLSX, JSON, JSONL)
# - ./rag 폴더 파일 변경 감지 (추가/수정 시 자동 재임베딩)
# - Vector Store 영속화 (한 번 임베딩한 데이터 재사용)
# =============================================================================

<<<<<<< HEAD
<<<<<<< HEAD
from utils.data_loader import get_rag_vector_store

=======
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    CSVLoader, 
    PyPDFLoader,
    UnstructuredExcelLoader,
    BSHTMLLoader
)
from langchain_community.document_loaders.base import BaseLoader

class JSONLLineLoader(BaseLoader):
    """
    JSONL(Line-delimited JSON) 파일을 한 줄씩 읽어서 Document로 변환하는 로더
    """
    def __init__(self, file_path, encoding='utf-8'):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        docs = []
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                for line in f:
                    if line.strip():
                        docs.append(Document(
                            page_content=line,
                            metadata={"source": self.file_path}
                        ))
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
        return docs

def dataloader(manager: VectorStoreManager):
    """
    LangChain의 DirectoryLoader를 사용하여 ./rag 폴더의 다양한 파일을 로딩합니다.
    """
    print("\n📥 LangChain DirectoryLoader를 통한 데이터 로딩 중...")
    
    rag_dir = "./rag"
    
    # 해당 폴더가 없으면 생성 (실습 편의용)
    if not os.path.exists(rag_dir):
        os.makedirs(rag_dir)
        print(f"   → {rag_dir} 폴더가 생성되었습니다. 파일을 넣어주세요.")

    # 1. 지원하는 파일 확장자와 로더 매핑
    # loader_map을 순회하며 확장자별로 DirectoryLoader를 설정합니다.
    loader_map = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".csv": CSVLoader,
        ".pdf": PyPDFLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".json": TextLoader,
        ".html": BSHTMLLoader,
        ".jsonl": JSONLLineLoader
    }
    
    all_documents = []
    
    for ext, loader_cls in loader_map.items():
        try:
            # 로더별 적절한 인자 설정
            loader_kwargs = {}
            if ext in [".txt", ".md", ".csv", ".json", ".jsonl"]:
                loader_kwargs["encoding"] = "utf-8"
            elif ext == ".html":
                loader_kwargs["open_encoding"] = "utf-8"

            # DirectoryLoader 설정: glob 패턴을 통해 특정 확장자 파일만 필터링
            # 💡 Windows 환경에서의 안정성을 위해 use_multithreading=False 설정을 권장합니다.
            loader = DirectoryLoader(
                path=rag_dir,
                glob=f"**/*{ext}", # 해당 확장자 파일 모두 찾기
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs, # 로더별 맞춤 인자 적용
                use_multithreading=False, # Windows 안정성을 위해 스레딩 비활성화
                silent_errors=True # 라이브러리 미설치 시 해당 확장자만 스킵
            )
            
            # 문서 로드
            docs = loader.load()
            if docs:
                all_documents.extend(docs)
                print(f"   → {ext} 파일 {len(docs)}개 로드 완료")
                
        except Exception as e:
            print(f"   ⚠️ {ext} 로더 경고: {str(e)[:50]}... (필요 라이브러리 확인 요망)")

    # 2. 로드된 문서가 있으면 Vector Store에 추가
    if all_documents:
        # manager.add_documents는 내부적으로 텍스트 분할(Chunking)을 수행합니다.
        manager.add_documents(all_documents)
        print(f"✅ 총 {len(all_documents)}개의 문서 조각이 Vector Store에 저장되었습니다.")
    else:
        # 데이터가 하나도 없는 경우 기본 텍스트라도 추가하여 동작 확인
        print("   ⚠️ 로딩된 문서가 없습니다. 기본 테스트 데이터를 적재합니다.")
        manager.add_texts(["LangGraph와 RAG 예제 데이터입니다."])
>>>>>>> e8ec7b919a6272dc13081ff5bf0c3aee7bd02f22

def get_vector_store() -> VectorStoreManager:
    """
    Vector Store를 초기화하고 데이터를 로드합니다.
    
    공통 데이터 로더 모듈을 사용하여:
    - ./rag 폴더의 모든 지원 파일을 자동 로딩
    - 파일 변경 감지로 필요시에만 재임베딩
    - 기존 임베딩 데이터 재사용
    """
    return get_rag_vector_store(collection_name="naive_rag")


# =============================================================================
# 🔧 3. 노드 함수 정의
# =============================================================================
#
# 각 노드 함수는:
# - 입력: state (현재 그래프 상태)
# - 출력: dict (업데이트할 필드와 값)
# =============================================================================

# 전역 Vector Store (한 번만 초기화)
_vector_store = None

def get_or_create_vector_store() -> VectorStoreManager:
    """
    Vector Store를 한 번만 초기화하고 재사용합니다.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store()
    return _vector_store

def retrieve(state: RAGState):
    """
    문서 검색 노드: 사용자 질문과 관련된 문서를 Vector Store에서 검색합니다.
    
    Args:
        state: 현재 RAG 상태 (question 필드 사용)
        
    Returns:
        dict: {"documents": 검색된 문서 리스트}
        
    💡 동작 원리:
       1. 사용자 질문을 벡터로 변환
       2. Vector Store에서 가장 유사한 벡터를 가진 문서들을 검색
       3. 상위 k개의 문서를 반환
    """
    print(f"� 검색 수행: {state['question']}")
    
    # Vector Store 가져오기 (전역 변수 사용)
    vs = get_or_create_vector_store()
    
    # 유사도 검색 수행 (상위 2개 문서 검색)
    # k: 검색할 문서 개수
    docs = vs.search(state["question"], k=2)
    
    # 검색 결과 로깅
    print(f"   → {len(docs)}개 문서 검색됨")
    for i, doc in enumerate(docs):
        print(f"   [{i+1}] {doc.page_content[:50]}...")
    
    # 상태 업데이트: documents 필드에 검색된 문서 저장
    return {"documents": docs}


def generate(state: RAGState):
    """
    답변 생성 노드: 검색된 문서를 참고하여 LLM이 답변을 생성합니다.
    
    Args:
        state: 현재 RAG 상태 (question, documents 필드 사용)
        
    Returns:
        dict: {"answer": 생성된 답변}
        
    💡 핵심 개념:
       - 검색된 문서들을 "컨텍스트"로 LLM에게 제공
       - LLM은 자신의 지식 + 컨텍스트를 조합하여 답변 생성
       - 이를 통해 LLM이 모르는 정보도 답변 가능
    """
    print("📝 답변 생성 중...")
    
    # -------------------------------------------------------------------------
    # Step 1: 컨텍스트 구성
    # -------------------------------------------------------------------------
    
    # 검색된 문서들의 내용을 하나의 문자열로 합침
    # 각 문서는 빈 줄로 구분
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    
    # -------------------------------------------------------------------------
    # Step 2: 프롬프트 템플릿 정의
    # -------------------------------------------------------------------------
    
    # {context}와 {question}은 나중에 실제 값으로 대체됨
    template = """다음 컨텍스트를 바탕으로 질문에 답변하세요.
    
    컨텍스트:
    {context}
    
    질문: {question}
    """
    
    # 템플릿을 ChatPromptTemplate 객체로 변환
    prompt = ChatPromptTemplate.from_template(template)
    
    # -------------------------------------------------------------------------
    # Step 3: 체인 구성 및 실행
    # -------------------------------------------------------------------------
    
    # LLM 인스턴스 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    # 체인 구성: 프롬프트 → LLM
    # LCEL (LangChain Expression Language) 문법: | 로 체인 연결
    chain = prompt | model
    
    # 체인 실행: 템플릿의 변수들에 실제 값을 넣어서 LLM 호출
    response = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    
    # 상태 업데이트: answer 필드에 생성된 답변 저장
    return {"answer": response.content}


# =============================================================================
# 🔀 4. 그래프 구성
# =============================================================================

def create_graph():
    """
    Naive RAG 그래프를 생성합니다.
    
    그래프 구조:
        START → retrieve → generate → END
        
    Returns:
        CompiledGraph: 실행 가능한 컴파일된 그래프
        
    💡 이 그래프는 선형 구조입니다 (분기/루프 없음)
       더 복잡한 패턴은 03, 04 예제에서 다룹니다.
    """
    # 그래프 빌더 생성 (RAGState를 상태 타입으로 사용)
    builder = StateGraph(RAGState)
    
    # -------------------------------------------------------------------------
    # 노드 추가
    # -------------------------------------------------------------------------
    
    # retrieve 노드: 문서 검색 담당
    builder.add_node("retrieve", retrieve)
    
    # generate 노드: 답변 생성 담당
    builder.add_node("generate", generate)
    
    # -------------------------------------------------------------------------
    # 엣지 연결 (선형 구조: 순서대로 실행)
    # -------------------------------------------------------------------------
    
    # START → retrieve: 시작하면 먼저 검색
    builder.add_edge(START, "retrieve")
    
    # retrieve → generate: 검색 후 답변 생성
    builder.add_edge("retrieve", "generate")
    
    # generate → END: 답변 생성 후 종료
    builder.add_edge("generate", END)
    
    # 그래프 컴파일 후 반환
    return builder.compile()


# =============================================================================
# ▶️ 5. 실행 함수
# =============================================================================

def run_rag(question: str):
    """
    RAG 파이프라인을 실행하여 질문에 답변합니다.
    
    Args:
        question: 사용자의 질문 문자열
    """
    # 그래프 생성
    app = create_graph()
    
    print(f"\n{'='*60}")
    print(f"🤖 질문: {question}")
    print('='*60)
    
    try:
        # 그래프 실행 (초기 상태: 질문만 설정)
        result = app.invoke({"question": question})
        
        # 결과 출력
        print(f"\n🤖 답변: {result['answer']}")
        
    except Exception as e:
        # 오류 상세 로깅
        log_llm_error(e)
        print("❌ 실행 중 오류가 발생했습니다.")


# =============================================================================
# 🚀 6. 메인 실행부 (CLI 인터페이스)
# =============================================================================

if __name__ == "__main__":
    # 프로그램 시작 메시지
    print("\n" + "="*60)
    print("📚 LangGraph Naive RAG Example")
    print("="*60)
    print("CLI 모드로 실행됩니다. 질문을 입력하세요.")
    print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.\n")
    
    # 무한 루프: 사용자가 종료할 때까지 질문-답변 반복
    while True:
        try:
            # 사용자 입력 받기
            question = input("🧑 질문을 입력하세요: ").strip()
            
            # 빈 입력 무시
            if not question:
                continue
            
            # 종료 명령어 확인
            if question.lower() in ("quit", "exit", "q"):
                print("👋 RAG Agent를 종료합니다. 안녕히 가세요!")
                break
            
            # RAG 실행
            run_rag(question)
            
        except KeyboardInterrupt:
            print("\n👋 RAG Agent를 종료합니다. (Ctrl+C)")
            break
        except EOFError:
            print("\n👋 RAG Agent를 종료합니다. (EOF)")
            break
