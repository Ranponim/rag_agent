# -*- coding: utf-8 -*-
"""
============================================================================
📚 03. Entity RAG 예제 - 엔티티 기반 병렬 검색
============================================================================

LangGraph의 병렬 실행(Parallel Execution) 기능을 활용하여,
엔티티 기반 검색과 의미론적 검색을 동시에 수행하고 결과를 병합하는 패턴을 학습합니다.

🎯 학습 목표:
    1. LangGraph의 병렬 노드 실행 (Fan-out / Fan-in) 패턴 구현
    2. LLM을 이용한 엔티티 추출 (Structured Output)
    3. 다중 검색 결과 병합 (Merge) 전략

💡 핵심 개념:
    - Fan-out: 하나의 노드에서 여러 노드로 동시에 분기
    - Fan-in: 여러 노드의 결과를 하나의 노드로 모음
    - Entity: 질문에서 추출한 핵심 키워드 (인물, 기술, 제품명 등)

그래프 구조:
                    ┌→ entity_search ─┐
    START → extract_entities ─┤                    ├→ merge → generate → END
                    └→ semantic_search ┘

실행 방법:
    python examples/03_entity_rag.py
    
    실행 후 CLI에서 질문을 입력하면 Entity RAG Agent가 응답합니다.
    종료: 'quit', 'exit', 또는 'q' 입력
"""

# =============================================================================
# 📦 필수 라이브러리 임포트
# =============================================================================

# Python 표준 라이브러리
import sys                              # 시스템 경로 조작용
import os                               # 환경변수 접근용
from pathlib import Path                # 파일 경로를 객체지향적으로 다루는 라이브러리
from typing import TypedDict, List      # 타입 힌트용

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# 🔗 LangChain 핵심 모듈 임포트
# -----------------------------------------------------------------------------

from langchain_openai import ChatOpenAI # LLM 모델 클래스
from langchain_core.documents import Document
# Document: 검색된 텍스트를 담는 표준 객체

from langchain_core.prompts import ChatPromptTemplate
# ChatPromptTemplate: 쿼리 분석 및 엔티티 추출용 프롬프트 템플릿

from langchain_core.output_parsers import JsonOutputParser
# JsonOutputParser: LLM의 JSON 형식 출력을 Python 딕셔너리로 변환
# 예: '{"entities": ["LangGraph"]}' → {"entities": ["LangGraph"]}

# -----------------------------------------------------------------------------
# 🔗 LangGraph 핵심 모듈 임포트
# -----------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END
# - StateGraph: 상태 기반 그래프 빌더
# - START/END: 시작점/종료점

# -----------------------------------------------------------------------------
# 🔗 프로젝트 내부 유틸리티 임포트
# -----------------------------------------------------------------------------

from utils.llm_factory import get_embeddings, log_llm_error
# LLM 및 임베딩 모델 생성

from utils.vector_store import VectorStoreManager
# 벡터 DB 관리


# =============================================================================
# 📋 1. State 정의
# =============================================================================
#
# Entity RAG에서는 병렬 검색을 위해 각 검색 결과를 별도 필드로 관리합니다.
# - entity_docs: 엔티티 기반 검색 결과 (키워드 매칭)
# - semantic_docs: 의미론적 검색 결과 (벡터 유사도)
# =============================================================================

class EntityRAGState(TypedDict):
    """
    Entity RAG 상태
    
    병렬 실행을 위해 각 검색 결과를 별도의 필드로 관리합니다.
    
    필드 설명:
    - question: 사용자 질문 (입력)
    - entities: 추출된 엔티티(키워드) 리스트
    - entity_docs: 엔티티 기반 검색 결과
    - semantic_docs: 의미론적(벡터) 검색 결과
    - merged_docs: 두 검색 결과를 병합한 문서 리스트
    - answer: 최종 답변 (출력)
    """
    question: str                    # 사용자 질문
    entities: List[str]              # 추출된 엔티티 이름 리스트
    entity_docs: List[Document]      # 엔티티 검색 결과 (Exact Match)
    semantic_docs: List[Document]    # 의미론적 검색 결과 (Vector Search)
    merged_docs: List[Document]      # 병합된 문서 리스트
    answer: str                      # 최종 답변


# =============================================================================
# 🗄️ 2. Vector Store 초기화 (공통 모듈 사용)
# =============================================================================

from utils.data_loader import get_rag_vector_store

def get_vector_store() -> VectorStoreManager:
    """Vector Store 초기화 및 데이터 로드"""
    return get_rag_vector_store(collection_name="rag_collection")


# =============================================================================
# 🔧 3. 노드 함수 정의
# =============================================================================

def extract_entities(state: EntityRAGState):
    """
    엔티티 추출 노드: LLM을 사용하여 질문에서 핵심 키워드를 추출합니다.
    
    Args:
        state: 현재 상태 (question 필드 사용)
        
    Returns:
        dict: {"entities": 추출된 엔티티 리스트}
        
    💡 동작 원리:
       1. LLM에게 "질문에서 엔티티를 추출해" 요청
       2. LLM이 JSON 형식으로 엔티티 리스트 반환
       3. JsonOutputParser가 Python 리스트로 변환
    """
    print(f"\n🏷️ 엔티티 추출 중: {state['question']}")
    
    # AI 모델 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    # JSON 출력을 유도하는 프롬프트
    # {{}}는 중괄호 리터럴 (f-string과 구분)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """질문에서 핵심 키워드(엔티티)를 추출하여 JSON 리스트로 반환하세요.
예시: {{"entities": ["Apple", "iPhone"]}}
질문: {question}"""),
    ])
    
    try:
        # 체인 구성: 프롬프트 → LLM → JSON 파서
        chain = prompt | model | JsonOutputParser()
        
        # 실행
        result = chain.invoke({"question": state["question"]})
        
        # 결과에서 entities 추출 (없으면 빈 리스트)
        entities = result.get("entities", [])
        print(f"   → 추출된 엔티티: {entities}")
        
        return {"entities": entities}
        
    except Exception as e:
        # 예외 발생 시 빈 리스트 반환 (graceful degradation)
        print(f"   → 추출 실패, 빈 리스트 반환: {e}")
        return {"entities": []}


def search_by_entity(state: EntityRAGState):
    """
    엔티티 기반 검색 노드 (병렬 실행 1)
    
    추출된 엔티티 각각에 대해 관련 문서를 검색합니다.
    
    💡 엔티티 검색 vs 의미론적 검색:
       - 엔티티 검색: 정확한 키워드 매칭 (LangGraph → LangGraph 관련 문서)
       - 의미론적 검색: 의미 유사도 기반 (LLM 도구 → AI 프레임워크 문서)
    """
    print("🔍 엔티티 검색 수행...")
    
    vs = get_vector_store()
    results = []
    
    # 각 엔티티에 대해 검색 수행
    for entity in state["entities"]:
        # 실제로는 메타데이터 필터링 사용 가능
        # 여기서는 단순 검색으로 시뮬레이션
        docs = vs.search(entity, k=1)  # 엔티티당 1개 문서
        results.extend(docs)
        print(f"   → '{entity}' 검색: {len(docs)}개 문서")

    return {"entity_docs": results}


def search_semantic(state: EntityRAGState):
    """
    의미론적 검색 노드 (병렬 실행 2)
    
    질문 전체의 의미를 기반으로 관련 문서를 검색합니다.
    """
    print("🔍 의미론적 검색 수행...")
    
    vs = get_vector_store()
    
    # 질문 전체로 벡터 유사도 검색
    docs = vs.search(state["question"], k=2)
    print(f"   → {len(docs)}개 문서 검색됨")
    
    return {"semantic_docs": docs}


def merge_results(state: EntityRAGState):
    """
    검색 결과 병합 노드 (Fan-in)
    
    엔티티 검색과 의미론적 검색 결과를 하나로 합칩니다.
    
    💡 병합 전략:
       1. 엔티티 검색 결과를 먼저 추가 (더 정확할 가능성)
       2. 의미론적 검색 결과 추가
       3. 중복 문서 제거 (같은 page_content는 한 번만)
    """
    print("🔄 검색 결과 병합 중...")
    
    # 중복 제거를 위한 집합
    seen = set()
    merged = []
    
    # 엔티티 검색 결과 우선 추가
    all_docs = state.get("entity_docs", []) + state.get("semantic_docs", [])
    
    for doc in all_docs:
        # page_content를 기준으로 중복 체크
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)

    print(f"   → 총 {len(merged)}개 문서 병합됨")
    
    return {"merged_docs": merged}


def generate_answer(state: EntityRAGState):
    """
    답변 생성 노드
    
    병합된 문서들을 컨텍스트로 사용하여 최종 답변을 생성합니다.
    """
    print("📝 답변 생성 중...")
    
    # 컨텍스트 구성
    context = "\n".join(d.page_content for d in state["merged_docs"])
    
    # LLM 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    response = model.invoke(f"컨텍스트: {context}\n\n질문: {state['question']}\n답변:")
    
    return {"answer": response.content}


# =============================================================================
# 🔀 4. 그래프 구성 (병렬 실행)
# =============================================================================

def create_graph():
    """
    Entity RAG 그래프 생성
    
    그래프 구조:
    
        START → extract_entities ─┬→ entity_search ──┬→ merge → generate → END
                                  └→ semantic_search ┘
                                  
    💡 병렬 실행 원리:
       - extract_entities에서 두 개의 엣지가 나감
       - LangGraph가 자동으로 두 노드를 동시 실행
       - merge 노드는 두 결과가 모두 도착해야 실행됨
    """
    builder = StateGraph(EntityRAGState)
    
    # -------------------------------------------------------------------------
    # 노드 추가
    # -------------------------------------------------------------------------
    
    builder.add_node("extract_entities", extract_entities)  # 엔티티 추출
    builder.add_node("entity_search", search_by_entity)     # 엔티티 검색
    builder.add_node("semantic_search", search_semantic)    # 의미론적 검색
    builder.add_node("merge", merge_results)                # 결과 병합
    builder.add_node("generate", generate_answer)           # 답변 생성
    
    # -------------------------------------------------------------------------
    # 엣지 연결
    # -------------------------------------------------------------------------
    
    # 시작 → 엔티티 추출
    builder.add_edge(START, "extract_entities")
    
    # 병렬 실행 (Fan-out): 엔티티 추출 후 두 검색 노드로 동시에 분기
    # 같은 source에서 두 개의 엣지를 추가하면 자동으로 병렬 실행!
    builder.add_edge("extract_entities", "entity_search")
    builder.add_edge("extract_entities", "semantic_search")
    
    # 병합 (Fan-in): 두 검색 노드가 모두 완료되면 merge 노드로 이동
    # 같은 destination으로 두 엣지가 들어오면 둘 다 완료 후 실행
    builder.add_edge("entity_search", "merge")
    builder.add_edge("semantic_search", "merge")
    
    # 이후는 선형 흐름
    builder.add_edge("merge", "generate")
    builder.add_edge("generate", END)
    
    return builder.compile()


# =============================================================================
# ▶️ 5. 실행 함수
# =============================================================================

def run_entity_rag(question: str):
    """
    Entity RAG 파이프라인을 실행하여 질문에 답변합니다.
    """
    app = create_graph()
    
    print(f"\n{'='*60}")
    print(f"🙋 질문: {question}")
    print('='*60)
    
    try:
        result = app.invoke({"question": question})
        print(f"\n🤖 답변: {result['answer']}")
        
    except Exception as e:
        log_llm_error(e)
        print("❌ 실행 중 오류가 발생했습니다.")


# =============================================================================
# 🚀 6. 메인 실행부 (CLI 인터페이스)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📚 LangGraph Entity RAG Example (Parallel Execution)")
    print("="*60)
    print("CLI 모드로 실행됩니다. 질문을 입력하세요.")
    print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.\n")
    
    while True:
        try:
            question = input("🙋 질문을 입력하세요: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ("quit", "exit", "q"):
                print("👋 Entity RAG Agent를 종료합니다. 안녕히 가세요!")
                break
            
            run_entity_rag(question)
            
        except KeyboardInterrupt:
            print("\n👋 Entity RAG Agent를 종료합니다. (Ctrl+C)")
            break
        except EOFError:
            print("\n👋 Entity RAG Agent를 종료합니다. (EOF)")
            break
