# -*- coding: utf-8 -*-
"""
02. Naive RAG 예제 - 기본 RAG 파이프라인 구현

LangGraph를 사용하여 가장 기본적인 검색-생성(Retrieve-Generate) 파이프라인을 구축합니다.
StateGraph를 활용하여 검색 결과와 생성된 답변을 상태로 관리하는 방법을 학습합니다.

학습 목표:
    1. RAG의 표준 파이프라인(Retrieve -> Generate) 구현
    2. 사용자 정의 State(TypedDict) 설계
    3. Vector Store 연동 및 검색 노드 구현

실행 방법:
    python examples/02_naive_rag.py
"""

import sys
from pathlib import Path
from typing import TypedDict, List, Annotated

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from config.settings import get_settings
from utils.llm_factory import get_llm, get_embeddings, log_llm_error
from utils.vector_store import VectorStoreManager


# =============================================================================
# 1. State 정의
# =============================================================================

class RAGState(TypedDict):
    """
    RAG 파이프라인 상태 정의
    
    필드 설명:
    - question: 사용자 질문 (Input)
    - documents: 검색된 문서 리스트 (Intermediate)
    - answer: 최종 답변 (Output)
    """
    question: str
    documents: List[Document]
    answer: str


# =============================================================================
# 2. Vector Store 및 데이터 준비
# =============================================================================

def get_vector_store() -> VectorStoreManager:
    """Vector Store 초기화 및 샘플 데이터 로드 (싱글톤 패턴)"""
    # 실제로는 별도 설정이나 DB에서 로드하겠지만, 여기서는 메모리에 생성
    embeddings = get_embeddings()
    manager = VectorStoreManager(embeddings=embeddings, collection_name="naive_rag")
    
    # 샘플 데이터가 비어있으면 추가
    # (주의: 실제 운영 환경에서는 매번 추가하지 않도록 체크 로직 필요)
    if True: # 간단한 예제를 위해 항상 로드 시도 (VectorStoreManager 내부에서 중복 처리 가정하거나 매번 재생성)
        texts = [
            "LangGraph는 LangChain 위에서 구축된 라이브러리로, 순환(Cyclic) 그래프를 지원합니다.",
            "RAG(Retrieval-Augmented Generation)는 외부 데이터를 검색하여 LLM의 맥락을 보강하는 기술입니다.",
            "LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.",
            "StateGraph는 LangGraph의 핵심 클래스로, 상태를 가진 노드들의 흐름을 정의합니다.",
        ]
        manager.add_texts(texts)

    return manager


# =============================================================================
# 3. 노드 함수 정의
# =============================================================================

def retrieve(state: RAGState):
    """문서 검색 노드"""
    print(f"\n🔍 검색 수행: {state['question']}")
    vs = get_vector_store()
    docs = vs.search(state["question"], k=2)
    return {"documents": docs}


def generate(state: RAGState):
    """답변 생성 노드"""
    print("📝 답변 생성 중...")
    
    # 컨텍스트 구성
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    
    # 프롬프트 템플릿
    template = """다음 컨텍스트를 바탕으로 질문에 답변하세요.
    
    컨텍스트:
    {context}
    
    질문: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 체인 구성
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    
    return {"answer": response.content}


# =============================================================================
# 4. 그래프 구성
# =============================================================================

def create_rag_graph():
    """Naive RAG 그래프 생성"""
    builder = StateGraph(RAGState)
    
    # 노드 추가
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    
    # 엣지 연결 (선형 구조)
    # START -> retrieve -> generate -> END
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    
    return builder.compile()


# =============================================================================
# 5. 실행 및 테스트
# =============================================================================

if __name__ == "__main__":
    print("\nLangGraph Naive RAG Example")
    
    graph = create_rag_graph()
    
    questions = [
        "LangGraph가 무엇인가요?",
        "RAG의 뜻은?",
    ]
    
    for q in questions:
        print(f"\n{'='*40}\n질문: {q}\n{'='*40}")
        try:
            result = graph.invoke({"question": q})
            print(f"\n🤖 답변: {result['answer']}")
        except Exception as e:
            log_llm_error(e)
