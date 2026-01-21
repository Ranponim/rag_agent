# -*- coding: utf-8 -*-
"""
02a. Rerank RAG - 검색 결과 재정렬

이 예제는 검색된 문서들을 LLM 기반으로 재정렬(Rerank)하여
더 관련성 높은 문서를 상위에 배치하는 RAG를 구현합니다.

학습 목표:
    1. 2단계 검색 전략 (Retrieve → Rerank)
    2. LLM 기반 관련성 점수 산정
    3. Cross-encoder 개념 이해
    4. Top-K 재선택

실행: python examples/02a_rerank_rag.py
"""

import sys
from pathlib import Path
from typing import TypedDict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from config.settings import get_settings
from utils.llm_factory import get_llm, get_embeddings
from utils.vector_store import VectorStoreManager


# =============================================================================
# 1. State 정의
# =============================================================================

class RerankRAGState(TypedDict):
    """Rerank RAG 상태"""
    question: str
    initial_documents: List[Document]    # 초기 검색 결과
    reranked_documents: List[Document]   # 재정렬된 문서
    rerank_scores: List[dict]            # 각 문서의 점수
    context: str
    answer: str


# =============================================================================
# 2. Vector Store 초기화
# =============================================================================

_rerank_vs: VectorStoreManager = None

def get_rerank_vs() -> VectorStoreManager:
    global _rerank_vs
    if _rerank_vs is None:
        print("📚 Rerank RAG Vector Store 초기화...")
        _rerank_vs = VectorStoreManager(
            embeddings=get_embeddings(),
            collection_name="rerank_rag",
            chunk_size=300,
        )
        samples = [
            "LangGraph는 상태 기반 에이전트를 위한 프레임워크입니다. StateGraph로 노드와 엣지를 정의하여 복잡한 워크플로우를 구현합니다.",
            "LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다. Chain 구조로 여러 컴포넌트를 연결합니다.",
            "RAG는 Retrieval-Augmented Generation의 약자로, 검색된 문서를 LLM의 컨텍스트로 제공하는 기법입니다.",
            "Vector Store는 임베딩 벡터를 저장하고 유사도 기반 검색을 수행하는 데이터베이스입니다. ChromaDB, Pinecone 등이 있습니다.",
            "Reranking은 초기 검색 결과를 재평가하여 순서를 재배치하는 기법입니다. Cross-encoder 모델을 주로 사용합니다.",
            "임베딩 모델은 텍스트를 고차원 벡터로 변환합니다. 의미적으로 유사한 텍스트는 벡터 공간에서 가까이 위치합니다.",
            "Python은 데이터 과학과 AI에서 가장 많이 사용되는 프로그래밍 언어입니다.",
            "FastAPI는 Python으로 API를 빠르게 개발할 수 있는 웹 프레임워크입니다.",
        ]
        _rerank_vs.add_texts(texts=samples)
        print(f"✅ {len(samples)}개 문서 추가")
    return _rerank_vs


# =============================================================================
# 3. 노드 함수
# =============================================================================

def retrieve_node(state: RerankRAGState) -> dict:
    """
    1단계: 초기 검색 (Over-fetch)
    
    Rerank를 위해 더 많은 문서를 검색합니다.
    최종적으로 필요한 것보다 2-3배 많이 가져옵니다.
    """
    print(f"\n🔍 [1단계] 초기 검색: '{state['question']}'")
    
    vs = get_rerank_vs()
    # 최종 필요 개수(3)보다 많이 검색 (6개)
    docs = vs.search(query=state["question"], k=6)
    
    print(f"   → {len(docs)}개 문서 검색됨")
    for i, doc in enumerate(docs):
        print(f"      [{i+1}] {doc.page_content[:50]}...")
    
    return {"initial_documents": docs}


def rerank_node(state: RerankRAGState) -> dict:
    """
    2단계: LLM 기반 Reranking
    
    각 문서의 관련성을 0-10 점수로 평가하고 재정렬합니다.
    Cross-encoder의 개념을 LLM으로 구현합니다.
    """
    print("\n📊 [2단계] Reranking...")
    
    llm = get_llm()
    
    # 각 문서에 대해 관련성 점수 산정
    prompt = ChatPromptTemplate.from_messages([
        ("system", """다음 문서가 질문에 얼마나 관련있는지 0-10 점수로 평가하세요.

점수 기준:
- 0-3: 관련 없음
- 4-6: 부분적으로 관련
- 7-10: 매우 관련있음

숫자만 답하세요."""),
        ("human", """질문: {question}

문서: {document}

관련성 점수 (0-10):"""),
    ])
    
    scored_docs = []
    
    for i, doc in enumerate(state["initial_documents"]):
        response = (prompt | llm).invoke({
            "question": state["question"],
            "document": doc.page_content
        })
        
        try:
            score = int(response.content.strip())
            score = max(0, min(10, score))  # 0-10 범위로 제한
        except:
            score = 5  # 파싱 실패 시 기본값
        
        scored_docs.append({
            "document": doc,
            "score": score,
            "original_rank": i + 1
        })
        
        print(f"   [{i+1}] 점수: {score}/10 - {doc.page_content[:40]}...")
    
    # 점수 기준으로 정렬 (내림차순)
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    # 상위 3개만 선택
    top_docs = scored_docs[:3]
    reranked = [item["document"] for item in top_docs]
    
    print(f"\n   → 재정렬 후 상위 3개:")
    for i, item in enumerate(top_docs):
        print(f"      [{i+1}] 점수: {item['score']}, 원래 순위: {item['original_rank']}")
    
    # 컨텍스트 생성
    context = "\n\n".join([
        f"[문서 {i+1}] {doc.page_content}" 
        for i, doc in enumerate(reranked)
    ])
    
    return {
        "reranked_documents": reranked,
        "rerank_scores": top_docs,
        "context": context
    }


def generate_node(state: RerankRAGState) -> dict:
    """3단계: 답변 생성"""
    print("\n💭 [3단계] 답변 생성...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """제공된 컨텍스트를 기반으로 질문에 답변하세요.
컨텍스트에 없는 정보는 추측하지 마세요.

컨텍스트:
{context}"""),
        ("human", "{question}"),
    ])
    
    response = (prompt | llm).invoke({
        "context": state["context"],
        "question": state["question"]
    })
    
    return {"answer": response.content}


# =============================================================================
# 4. 그래프 생성
# =============================================================================

def create_rerank_rag_graph():
    """
    Rerank RAG 그래프
    
    구조: START → retrieve → rerank → generate → END
    """
    graph = StateGraph(RerankRAGState)
    
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    
    print("✅ Rerank RAG 그래프 컴파일 완료!")
    return graph.compile()


# =============================================================================
# 5. 실행
# =============================================================================

def run_rerank_rag(question: str) -> str:
    graph = create_rerank_rag_graph()
    
    initial_state = {
        "question": question,
        "initial_documents": [],
        "reranked_documents": [],
        "rerank_scores": [],
        "context": "",
        "answer": ""
    }
    
    print(f"\n{'='*60}")
    print(f"🙋 질문: {question}")
    print('='*60)
    
    result = graph.invoke(initial_state)
    
    print(f"\n🤖 답변:\n{result['answer']}")
    print('='*60)
    
    return result["answer"]


if __name__ == "__main__":
    from utils.llm_factory import log_llm_error
    
    print("\n" + "="*60)
    print("Rerank RAG 예제")
    print("="*60)
    
    queries = [
        "LangGraph가 뭐야?",
        "RAG에서 Reranking은 왜 필요해?",
    ]
    
    for query in queries:
        try:
            run_rerank_rag(query)
        except Exception as e:
            log_llm_error(e)
            print(f"❌ 오류: {e}")
        print()
