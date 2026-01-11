# -*- coding: utf-8 -*-
"""
02. Naive RAG 예제 - 기본 RAG 파이프라인 구현

이 예제는 LangGraph를 사용한 가장 기본적인 RAG 파이프라인을 구현합니다.
문서를 Vector Store에 저장하고, 검색된 문서를 기반으로 답변을 생성합니다.

학습 목표:
    1. RAG의 기본 동작 원리 이해
    2. Vector Store와 Retriever 연동 방법
    3. 검색 → 생성 파이프라인 구현
    4. LangGraph에서 RAG 구현 패턴 학습

실행 방법:
    python examples/02_naive_rag.py

필수 환경 변수:
    OPENAI_API_KEY: OpenAI API 키
"""

import sys
from pathlib import Path
from typing import TypedDict, List, Annotated
from operator import add

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from config.settings import get_settings
from utils.llm_factory import get_llm, get_embeddings
from utils.vector_store import VectorStoreManager


# =============================================================================
# 1. State 정의
# =============================================================================

class RAGState(TypedDict):
    """
    RAG 파이프라인의 상태를 정의합니다.
    
    TypedDict를 사용하여 상태의 스키마를 명확하게 정의합니다.
    각 필드는 그래프를 통해 전달되는 데이터를 나타냅니다.
    
    Attributes:
        question: 사용자의 질문
        context: 검색된 문서 내용 (문자열로 결합)
        documents: 검색된 Document 객체 리스트
        answer: 생성된 답변
    """
    question: str                    # 사용자 질문
    context: str                     # 검색된 컨텍스트 (문자열)
    documents: List[Document]        # 검색된 문서 리스트
    answer: str                      # 최종 답변


# =============================================================================
# 2. Vector Store 초기화
# =============================================================================

def initialize_vector_store() -> VectorStoreManager:
    """
    Vector Store를 초기화하고 샘플 문서를 로드합니다.
    
    Returns:
        VectorStoreManager: 초기화된 Vector Store 매니저
    """
    print("📚 Vector Store 초기화 중...")
    
    # 임베딩 모델 생성
    embeddings = get_embeddings()
    
    # Vector Store 매니저 생성
    manager = VectorStoreManager(
        embeddings=embeddings,
        collection_name="naive_rag_example",
        chunk_size=500,
        chunk_overlap=100,
    )
    
    # 샘플 문서 로드
    sample_file = Path(__file__).parent.parent / "data" / "sample_documents.txt"
    
    if sample_file.exists():
        manager.load_from_file(str(sample_file))
        print(f"✅ 샘플 문서 로드 완료: {sample_file}")
    else:
        # 샘플 문서가 없으면 기본 문서 추가
        sample_texts = [
            "LangGraph는 LangChain 팀에서 개발한 라이브러리로, 상태를 가진 다중 행위자 애플리케이션을 구축합니다.",
            "RAG는 Retrieval-Augmented Generation의 약자로, 검색 증강 생성을 의미합니다.",
            "Vector Store는 임베딩 벡터를 저장하고 유사도 검색을 수행하는 데이터베이스입니다.",
            "LangGraph의 핵심 개념은 State, Node, Edge입니다.",
            "StateGraph는 LangGraph에서 그래프를 구성하는 빌더 클래스입니다.",
        ]
        manager.add_texts(
            texts=sample_texts,
            metadatas=[{"source": "sample"} for _ in sample_texts]
        )
        print("✅ 기본 샘플 문서 추가 완료")
    
    return manager


# 전역 Vector Store 매니저 (한 번만 초기화)
_vector_store_manager: VectorStoreManager = None


def get_vector_store() -> VectorStoreManager:
    """Vector Store 매니저 싱글톤 반환"""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = initialize_vector_store()
    return _vector_store_manager


# =============================================================================
# 3. 노드 함수 정의
# =============================================================================

def retrieve_node(state: RAGState) -> dict:
    """
    검색 노드: 사용자 질문과 관련된 문서를 검색합니다.
    
    이 노드는 RAG 파이프라인의 "R" (Retrieval) 단계입니다.
    Vector Store에서 질문과 유사한 문서를 검색합니다.
    
    Args:
        state: 현재 RAG 상태
    
    Returns:
        dict: 업데이트된 상태 (documents, context 포함)
    
    Flow:
        1. 사용자 질문 추출
        2. Vector Store에서 유사 문서 검색
        3. 검색된 문서를 컨텍스트 문자열로 변환
    """
    print(f"\n🔍 검색 중: '{state['question']}'")
    
    # Vector Store에서 문서 검색
    manager = get_vector_store()
    documents = manager.search(
        query=state["question"],
        k=3  # 상위 3개 문서 검색
    )
    
    print(f"   → {len(documents)}개 문서 발견")
    
    # 문서 내용을 컨텍스트 문자열로 결합
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[문서 {i}]\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    return {
        "documents": documents,
        "context": context,
    }


def generate_node(state: RAGState) -> dict:
    """
    생성 노드: 검색된 문서를 바탕으로 답변을 생성합니다.
    
    이 노드는 RAG 파이프라인의 "G" (Generation) 단계입니다.
    LLM을 사용하여 컨텍스트 기반 답변을 생성합니다.
    
    Args:
        state: 현재 RAG 상태 (context 포함)
    
    Returns:
        dict: 업데이트된 상태 (answer 포함)
    
    Flow:
        1. 프롬프트 템플릿 구성
        2. 컨텍스트와 질문을 프롬프트에 포함
        3. LLM 호출하여 답변 생성
    """
    print("\n💭 답변 생성 중...")
    
    # LLM 생성
    llm = get_llm()
    
    # RAG 프롬프트 템플릿
    # 컨텍스트를 기반으로 질문에 답변하도록 지시합니다
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 도움이 되는 AI 어시스턴트입니다.
아래 제공된 컨텍스트를 기반으로 사용자의 질문에 답변하세요.

중요:
- 컨텍스트에 있는 정보만 사용하세요
- 컨텍스트에 답이 없으면 "제공된 정보에서 답을 찾을 수 없습니다"라고 말하세요
- 답변은 명확하고 간결하게 작성하세요

컨텍스트:
{context}"""),
        ("human", "{question}"),
    ])
    
    # 프롬프트 구성 및 LLM 호출
    chain = prompt | llm
    
    response = chain.invoke({
        "context": state["context"],
        "question": state["question"],
    })
    
    print("   → 답변 생성 완료")
    
    return {"answer": response.content}


# =============================================================================
# 4. RAG 그래프 생성
# =============================================================================

def create_rag_graph():
    """
    Naive RAG 그래프를 생성합니다.
    
    그래프 구조 (단순 선형 파이프라인):
        START → retrieve → generate → END
    
    Returns:
        CompiledGraph: 컴파일된 RAG 그래프
    
    Note:
        Naive RAG는 가장 단순한 형태의 RAG입니다:
        1. 질문을 받음
        2. 관련 문서 검색
        3. 검색된 문서로 답변 생성
        
        단점:
        - 검색 결과의 품질을 검증하지 않음
        - 답변의 정확성을 확인하지 않음
        - 반복적인 검색이 불가능
    """
    # StateGraph 생성 (RAGState 사용)
    graph = StateGraph(RAGState)
    
    # ----- 노드 추가 -----
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    # ----- 엣지 추가 -----
    # 단순 선형 파이프라인: START → retrieve → generate → END
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    # 그래프 컴파일
    compiled_graph = graph.compile()
    
    print("✅ Naive RAG 그래프 컴파일 완료!")
    return compiled_graph


# =============================================================================
# 5. RAG 실행
# =============================================================================

def run_rag(question: str) -> str:
    """
    RAG 파이프라인을 실행합니다.
    
    Args:
        question: 사용자 질문
    
    Returns:
        str: 생성된 답변
    """
    # 그래프 생성
    graph = create_rag_graph()
    
    # 초기 상태 설정
    initial_state = {
        "question": question,
        "context": "",
        "documents": [],
        "answer": "",
    }
    
    print(f"\n{'='*60}")
    print(f"🙋 질문: {question}")
    print('='*60)
    
    # 그래프 실행
    result = graph.invoke(initial_state)
    
    # 결과 출력
    print(f"\n📚 검색된 문서 수: {len(result['documents'])}")
    print(f"\n🤖 답변:\n{result['answer']}")
    print('='*60)
    
    return result["answer"]


def run_rag_with_stream(question: str):
    """
    스트리밍 모드로 RAG를 실행합니다.
    
    각 노드의 실행 과정을 실시간으로 확인할 수 있습니다.
    
    Args:
        question: 사용자 질문
    """
    graph = create_rag_graph()
    
    initial_state = {
        "question": question,
        "context": "",
        "documents": [],
        "answer": "",
    }
    
    print(f"\n{'='*60}")
    print(f"🙋 질문: {question}")
    print('='*60)
    
    # stream()으로 각 단계 확인
    for event in graph.stream(initial_state, stream_mode="updates"):
        # event는 {노드명: 상태 업데이트} 형태
        for node_name, updates in event.items():
            print(f"\n📍 노드: {node_name}")
            
            if "documents" in updates:
                print(f"   → 검색된 문서: {len(updates['documents'])}개")
            
            if "context" in updates:
                preview = updates["context"][:100] + "..." if len(updates.get("context", "")) > 100 else updates.get("context", "")
                print(f"   → 컨텍스트 미리보기: {preview}")
            
            if "answer" in updates:
                print(f"   → 답변: {updates['answer'][:200]}...")


# =============================================================================
# 6. 시각화 (선택)
# =============================================================================

def visualize_graph():
    """
    RAG 그래프 구조를 시각화합니다.
    
    Mermaid 다이어그램 형식으로 그래프 구조를 출력합니다.
    """
    print("\n📊 RAG 그래프 구조 (Mermaid)")
    print("```mermaid")
    print("graph TD")
    print("    START((START)) --> retrieve[검색 노드]")
    print("    retrieve --> generate[생성 노드]")
    print("    generate --> END((END))")
    print("")
    print("    subgraph State")
    print("        Q[question]")
    print("        C[context]")
    print("        D[documents]")
    print("        A[answer]")
    print("    end")
    print("```")


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Naive RAG 예제 - 기본 RAG 파이프라인")
    print("="*60)
    
    # 설정 확인
    settings = get_settings()
    if not settings.validate_openai_key():
        print("\n⚠️ OpenAI API 키가 설정되지 않았습니다.")
        print("📝 .env 파일에 OPENAI_API_KEY를 설정해주세요.")
        sys.exit(1)
    
    # 그래프 시각화
    visualize_graph()
    
    # 테스트 쿼리 실행
    test_queries = [
        "LangGraph란 무엇인가요?",
        "RAG의 기본 구성 요소는 무엇인가요?",
        "StateGraph는 어떤 역할을 하나요?",
    ]
    
    for query in test_queries:
        try:
            run_rag(query)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n")
