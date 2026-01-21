# -*- coding: utf-8 -*-
"""
02b. Query Transform RAG - 쿼리 변환 RAG

이 예제는 사용자 쿼리를 변환하여 검색 효율을 높이는 RAG를 구현합니다.
HyDE(Hypothetical Document Embeddings)와 Multi-Query 기법을 사용합니다.

학습 목표:
    1. HyDE: 가상 문서 생성 후 검색
    2. Multi-Query: 쿼리를 여러 변형으로 확장
    3. 쿼리 분해: 복잡한 질문을 단순한 질문들로 분해
    4. 결과 퓨전

실행: python examples/02b_query_transform_rag.py
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

class QueryTransformState(TypedDict):
    """Query Transform RAG 상태"""
    original_question: str               # 원본 질문
    hyde_document: str                   # HyDE로 생성된 가상 문서
    multi_queries: List[str]             # Multi-Query 변형들
    hyde_results: List[Document]         # HyDE 검색 결과
    multi_query_results: List[Document]  # Multi-Query 검색 결과
    merged_documents: List[Document]     # 병합된 문서
    context: str
    answer: str


# =============================================================================
# 2. Vector Store 초기화
# =============================================================================

_qt_vs: VectorStoreManager = None

def get_qt_vs() -> VectorStoreManager:
    global _qt_vs
    if _qt_vs is None:
        print("📚 Query Transform Vector Store 초기화...")
        _qt_vs = VectorStoreManager(
            embeddings=get_embeddings(),
            collection_name="query_transform_rag",
            chunk_size=300,
        )
        samples = [
            "LangGraph는 LangChain 팀이 개발한 상태 기반 에이전트 프레임워크입니다. StateGraph를 사용하여 노드와 엣지를 정의합니다.",
            "RAG(Retrieval-Augmented Generation)는 검색 증강 생성 기법으로, LLM에게 관련 문서를 컨텍스트로 제공합니다.",
            "HyDE(Hypothetical Document Embeddings)는 질문에 대한 가상의 답변을 먼저 생성하고, 그 답변으로 검색하는 기법입니다.",
            "Multi-Query는 하나의 질문을 여러 관점에서 재작성하여 검색 범위를 넓히는 기법입니다.",
            "임베딩은 텍스트를 고차원 벡터로 변환하는 과정입니다. 유사한 의미를 가진 텍스트는 유사한 벡터를 갖습니다.",
            "Vector Store는 벡터 데이터베이스로, 임베딩된 문서를 저장하고 유사도 검색을 수행합니다.",
            "Query Decomposition은 복잡한 질문을 여러 단순한 질문으로 분해하는 기법입니다.",
            "Reciprocal Rank Fusion은 여러 검색 결과를 통합할 때 순위를 고려하여 병합하는 알고리즘입니다.",
        ]
        _qt_vs.add_texts(texts=samples)
        print(f"✅ {len(samples)}개 문서 추가")
    return _qt_vs


# =============================================================================
# 3. 노드 함수
# =============================================================================

def generate_hyde_document(state: QueryTransformState) -> dict:
    """
    HyDE: 가상 문서 생성
    
    질문에 대한 가상의 답변을 먼저 생성합니다.
    이 답변은 실제 문서와 유사한 어휘를 포함할 가능성이 높아
    임베딩 기반 검색 효율이 높아집니다.
    """
    print(f"\n🔮 [HyDE] 가상 문서 생성 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 질문에 대해 상세한 설명을 제공하는 전문가입니다.
다음 질문에 대해 마치 교과서나 문서에 있을 법한 상세한 답변을 작성하세요.
실제로 정확한지 모르더라도, 가능한 전문적인 어휘를 사용하세요."""),
        ("human", "{question}"),
    ])
    
    response = (prompt | llm).invoke({"question": state["original_question"]})
    hyde_doc = response.content
    
    print(f"   → 가상 문서: {hyde_doc[:100]}...")
    
    return {"hyde_document": hyde_doc}


def generate_multi_queries(state: QueryTransformState) -> dict:
    """
    Multi-Query: 쿼리 변형 생성
    
    원본 질문을 다양한 관점에서 재작성하여
    검색 범위를 넓힙니다.
    """
    print(f"\n🔄 [Multi-Query] 쿼리 변형 생성 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 검색 쿼리 전문가입니다.
주어진 질문을 3가지 다른 관점에서 재작성하세요.
각 질문은 같은 정보를 찾지만 다른 표현을 사용해야 합니다.

형식:
1. [첫 번째 변형]
2. [두 번째 변형]
3. [세 번째 변형]"""),
        ("human", "원본 질문: {question}"),
    ])
    
    response = (prompt | llm).invoke({"question": state["original_question"]})
    
    # 응답에서 질문들 추출
    lines = response.content.strip().split("\n")
    queries = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # 번호나 대시 제거
            query = line.lstrip("0123456789.-) ").strip()
            if query:
                queries.append(query)
    
    # 원본 질문도 포함
    queries = [state["original_question"]] + queries[:3]
    
    print(f"   → 쿼리 변형들:")
    for i, q in enumerate(queries):
        print(f"      [{i+1}] {q}")
    
    return {"multi_queries": queries}


def search_with_hyde(state: QueryTransformState) -> dict:
    """HyDE 문서로 검색"""
    print(f"\n🔍 [HyDE 검색] 가상 문서로 검색 중...")
    
    vs = get_qt_vs()
    docs = vs.search(query=state["hyde_document"], k=3)
    
    print(f"   → {len(docs)}개 문서 검색됨")
    
    return {"hyde_results": docs}


def search_with_multi_queries(state: QueryTransformState) -> dict:
    """Multi-Query로 검색"""
    print(f"\n🔍 [Multi-Query 검색] 여러 쿼리로 검색 중...")
    
    vs = get_qt_vs()
    all_docs = []
    seen_contents = set()
    
    for i, query in enumerate(state["multi_queries"]):
        docs = vs.search(query=query, k=2)
        for doc in docs:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)
        print(f"   쿼리 [{i+1}]: {len(docs)}개")
    
    print(f"   → 총 {len(all_docs)}개 고유 문서")
    
    return {"multi_query_results": all_docs}


def merge_results(state: QueryTransformState) -> dict:
    """
    결과 병합 (Reciprocal Rank Fusion 개념 적용)
    
    HyDE와 Multi-Query 결과를 병합하고 중복 제거합니다.
    """
    print(f"\n🔀 [병합] 결과 통합 중...")
    
    # 두 결과 병합 (중복 제거)
    seen = set()
    merged = []
    
    # HyDE 결과 먼저 (보통 더 정확)
    for doc in state.get("hyde_results", []):
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    
    # Multi-Query 결과 추가
    for doc in state.get("multi_query_results", []):
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    
    # 최대 5개로 제한
    merged = merged[:5]
    
    context = "\n\n".join([
        f"[문서 {i+1}] {doc.page_content}"
        for i, doc in enumerate(merged)
    ])
    
    print(f"   → 최종 {len(merged)}개 문서")
    
    return {"merged_documents": merged, "context": context}


def generate_answer(state: QueryTransformState) -> dict:
    """답변 생성"""
    print(f"\n💭 [생성] 답변 생성 중...")
    
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """컨텍스트를 기반으로 질문에 답변하세요.

컨텍스트:
{context}"""),
        ("human", "{question}"),
    ])
    
    response = (prompt | llm).invoke({
        "context": state["context"],
        "question": state["original_question"]
    })
    
    return {"answer": response.content}


# =============================================================================
# 4. 그래프 생성
# =============================================================================

def create_query_transform_rag():
    """
    Query Transform RAG 그래프
    
    구조:
        START → generate_hyde ─────────→ search_hyde ──────┐
              └→ generate_multi_queries → search_multi ────┴→ merge → generate → END
    """
    graph = StateGraph(QueryTransformState)
    
    # 노드 추가
    graph.add_node("generate_hyde", generate_hyde_document)
    graph.add_node("generate_multi_queries", generate_multi_queries)
    graph.add_node("search_hyde", search_with_hyde)
    graph.add_node("search_multi", search_with_multi_queries)
    graph.add_node("merge", merge_results)
    graph.add_node("generate", generate_answer)
    
    # 엣지 (병렬 쿼리 변환)
    graph.add_edge(START, "generate_hyde")
    graph.add_edge(START, "generate_multi_queries")
    graph.add_edge("generate_hyde", "search_hyde")
    graph.add_edge("generate_multi_queries", "search_multi")
    graph.add_edge("search_hyde", "merge")
    graph.add_edge("search_multi", "merge")
    graph.add_edge("merge", "generate")
    graph.add_edge("generate", END)
    
    print("✅ Query Transform RAG 컴파일 완료!")
    return graph.compile()


# =============================================================================
# 5. 실행
# =============================================================================

def run_query_transform_rag(question: str) -> str:
    graph = create_query_transform_rag()
    
    initial_state = {
        "original_question": question,
        "hyde_document": "",
        "multi_queries": [],
        "hyde_results": [],
        "multi_query_results": [],
        "merged_documents": [],
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
    print("Query Transform RAG 예제")
    print("="*60)
    
    queries = [
        "HyDE가 뭐야?",
        "RAG에서 쿼리 변환은 어떤 종류가 있어?",
    ]
    
    for query in queries:
        try:
            run_query_transform_rag(query)
        except Exception as e:
            log_llm_error(e)
            print(f"❌ 오류: {e}")
        print()
