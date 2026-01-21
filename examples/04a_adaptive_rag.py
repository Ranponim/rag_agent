# -*- coding: utf-8 -*-
"""
04a. Adaptive RAG - 쿼리 복잡도에 따른 적응형 RAG

이 예제는 쿼리의 복잡도를 분석하여 적절한 RAG 전략을 
동적으로 선택하는 Adaptive RAG를 구현합니다.

학습 목표:
    1. 쿼리 복잡도 분류 (단순/중간/복잡)
    2. 전략별 다른 파이프라인 실행
    3. 동적 라우팅
    4. 비용-품질 트레이드오프

실행: python examples/04a_adaptive_rag.py
"""

import sys
from pathlib import Path
from typing import TypedDict, List, Literal

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

class AdaptiveRAGState(TypedDict):
    """Adaptive RAG 상태"""
    question: str
    query_complexity: str            # "simple" | "moderate" | "complex"
    strategy_used: str               # 사용된 전략
    documents: List[Document]
    context: str
    answer: str


# =============================================================================
# 2. Vector Store 초기화
# =============================================================================

_adaptive_vs: VectorStoreManager = None

def get_adaptive_vs() -> VectorStoreManager:
    global _adaptive_vs
    if _adaptive_vs is None:
        print("📚 Adaptive RAG Vector Store 초기화...")
        _adaptive_vs = VectorStoreManager(
            embeddings=get_embeddings(),
            collection_name="adaptive_rag",
        )
        samples = [
            "LangGraph는 상태 기반 에이전트를 구축하기 위한 프레임워크입니다.",
            "RAG는 Retrieval-Augmented Generation의 약자로, 검색 증강 생성입니다.",
            "Adaptive RAG는 쿼리 복잡도에 따라 다른 전략을 사용합니다.",
            "Self-RAG는 LLM이 검색 필요성과 답변 품질을 스스로 평가합니다.",
            "Vector Store는 임베딩 벡터를 저장하고 유사도 검색을 수행합니다.",
            "임베딩은 텍스트를 고차원 벡터로 변환하는 과정입니다.",
        ]
        _adaptive_vs.add_texts(texts=samples)
        print(f"✅ {len(samples)}개 문서 추가")
    return _adaptive_vs


# =============================================================================
# 3. 쿼리 복잡도 분류
# =============================================================================

def classify_query_node(state: AdaptiveRAGState) -> dict:
    """
    쿼리 복잡도 분류
    
    분류 기준:
    - simple: 정의, 단순 사실 질문 → 검색 없이 직접 답변
    - moderate: 일반적인 정보 질문 → 기본 RAG
    - complex: 분석, 비교, 다단계 추론 → 고급 RAG
    """
    print(f"\n🔍 [분류] 쿼리 복잡도 분석 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """질문의 복잡도를 분류하세요.

- simple: 간단한 정의, 단순 사실 질문 (예: "RAG가 뭐야?")
- moderate: 일반적인 정보 요청 (예: "RAG의 장점은?")
- complex: 분석, 비교, 다단계 추론 필요 (예: "RAG와 Fine-tuning을 비교해서 언제 뭘 써야 할지 설명해줘")

"simple", "moderate", "complex" 중 하나만 답하세요."""),
        ("human", "질문: {question}"),
    ])
    
    response = (prompt | llm).invoke({"question": state["question"]})
    
    content = response.content.lower().strip()
    if "complex" in content:
        complexity = "complex"
    elif "moderate" in content:
        complexity = "moderate"
    else:
        complexity = "simple"
    
    print(f"   → 복잡도: {complexity}")
    
    return {"query_complexity": complexity}


# =============================================================================
# 4. 전략별 노드
# =============================================================================

def simple_strategy_node(state: AdaptiveRAGState) -> dict:
    """
    Simple 전략: 검색 없이 직접 답변
    
    간단한 질문은 LLM의 기본 지식으로 충분히 답변 가능.
    검색 비용을 절약합니다.
    """
    print("\n⚡ [Simple 전략] 직접 답변 생성...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "간단하고 명확하게 답변하세요."),
        ("human", "{question}"),
    ])
    
    response = (prompt | llm).invoke({"question": state["question"]})
    
    return {
        "strategy_used": "simple (직접 답변)",
        "answer": response.content
    }


def moderate_strategy_node(state: AdaptiveRAGState) -> dict:
    """
    Moderate 전략: 기본 RAG
    
    검색 + 생성의 표준 RAG 파이프라인.
    """
    print("\n📚 [Moderate 전략] 기본 RAG 실행...")
    
    # 검색
    vs = get_adaptive_vs()
    docs = vs.search(query=state["question"], k=3)
    
    context = "\n".join([doc.page_content for doc in docs])
    print(f"   → {len(docs)}개 문서 검색됨")
    
    # 생성
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """컨텍스트를 기반으로 답변하세요.

컨텍스트:
{context}"""),
        ("human", "{question}"),
    ])
    
    response = (prompt | llm).invoke({
        "context": context,
        "question": state["question"]
    })
    
    return {
        "strategy_used": "moderate (기본 RAG)",
        "documents": docs,
        "context": context,
        "answer": response.content
    }


def complex_strategy_node(state: AdaptiveRAGState) -> dict:
    """
    Complex 전략: 고급 RAG (다단계 추론)
    
    1. 질문 분해
    2. 각 하위 질문에 대해 검색
    3. 통합 답변 생성
    """
    print("\n🔬 [Complex 전략] 고급 RAG 실행...")
    
    llm = get_llm()
    
    # 1단계: 질문 분해
    print("   [1/3] 질문 분해...")
    decompose_prompt = ChatPromptTemplate.from_messages([
        ("system", """복잡한 질문을 2-3개의 하위 질문으로 분해하세요.
각 하위 질문은 한 줄씩 작성하세요."""),
        ("human", "{question}"),
    ])
    
    sub_questions_response = (decompose_prompt | llm).invoke({
        "question": state["question"]
    })
    sub_questions = [q.strip() for q in sub_questions_response.content.strip().split("\n") if q.strip()][:3]
    
    print(f"      → 하위 질문: {sub_questions}")
    
    # 2단계: 각 질문에 대해 검색
    print("   [2/3] 하위 질문별 검색...")
    vs = get_adaptive_vs()
    all_docs = []
    seen = set()
    
    for sq in sub_questions:
        docs = vs.search(query=sq, k=2)
        for doc in docs:
            if doc.page_content not in seen:
                all_docs.append(doc)
                seen.add(doc.page_content)
    
    context = "\n\n".join([doc.page_content for doc in all_docs])
    print(f"      → 총 {len(all_docs)}개 문서")
    
    # 3단계: 통합 답변 생성
    print("   [3/3] 통합 답변 생성...")
    synthesize_prompt = ChatPromptTemplate.from_messages([
        ("system", """다음 하위 질문들과 컨텍스트를 바탕으로 
원본 질문에 대한 종합적인 답변을 작성하세요.

하위 질문들: {sub_questions}

컨텍스트:
{context}"""),
        ("human", "원본 질문: {question}"),
    ])
    
    response = (synthesize_prompt | llm).invoke({
        "question": state["question"],
        "sub_questions": sub_questions,
        "context": context
    })
    
    return {
        "strategy_used": f"complex (다단계 RAG, 하위질문: {len(sub_questions)}개)",
        "documents": all_docs,
        "context": context,
        "answer": response.content
    }


# =============================================================================
# 5. 라우터
# =============================================================================

def route_by_complexity(state: AdaptiveRAGState) -> Literal["simple", "moderate", "complex"]:
    """복잡도에 따라 전략 라우팅"""
    complexity = state.get("query_complexity", "moderate")
    print(f"🔀 라우팅: {complexity} 전략으로 이동")
    return complexity


# =============================================================================
# 6. 그래프 생성
# =============================================================================

def create_adaptive_rag_graph():
    """
    Adaptive RAG 그래프
    
    구조:
        START → classify → (simple | moderate | complex) → END
    """
    graph = StateGraph(AdaptiveRAGState)
    
    graph.add_node("classify", classify_query_node)
    graph.add_node("simple", simple_strategy_node)
    graph.add_node("moderate", moderate_strategy_node)
    graph.add_node("complex", complex_strategy_node)
    
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_complexity,
        {
            "simple": "simple",
            "moderate": "moderate",
            "complex": "complex"
        }
    )
    graph.add_edge("simple", END)
    graph.add_edge("moderate", END)
    graph.add_edge("complex", END)
    
    print("✅ Adaptive RAG 컴파일 완료!")
    return graph.compile()


# =============================================================================
# 7. 실행
# =============================================================================

def run_adaptive_rag(question: str) -> str:
    graph = create_adaptive_rag_graph()
    
    initial_state = {
        "question": question,
        "query_complexity": "",
        "strategy_used": "",
        "documents": [],
        "context": "",
        "answer": ""
    }
    
    print(f"\n{'='*60}")
    print(f"🙋 질문: {question}")
    print('='*60)
    
    result = graph.invoke(initial_state)
    
    print(f"\n📊 사용된 전략: {result['strategy_used']}")
    print(f"\n🤖 답변:\n{result['answer']}")
    print('='*60)
    
    return result["answer"]


if __name__ == "__main__":
    from utils.llm_factory import log_llm_error
    
    print("\n" + "="*60)
    print("Adaptive RAG 예제")
    print("="*60)
    
    queries = [
        "RAG가 뭐야?",                              # simple
        "LangGraph의 주요 특징은?",                  # moderate
        "RAG와 Fine-tuning을 비교하고 각각 언제 사용해야 할지 분석해줘",  # complex
    ]
    
    for query in queries:
        try:
            run_adaptive_rag(query)
        except Exception as e:
            log_llm_error(e)
            print(f"❌ 오류: {e}")
        print()
