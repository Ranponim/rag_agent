# -*- coding: utf-8 -*-
"""
04. Advanced RAG 예제 - Self-RAG & Corrective RAG 구현

고급 RAG 패턴으로 검색 결과 평가, 관련성 검증, 자기 수정 루프를 구현합니다.

학습 목표:
    1. 조건부 분기를 활용한 적응형 RAG
    2. 문서 관련성 평가 (Grading)
    3. 답변 품질 검증 (Hallucination Check)
    4. 자기 수정 루프 구현

실행: python examples/04_advanced_rag.py
"""

import sys
from pathlib import Path
from typing import TypedDict, List, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END

from config.settings import get_settings
from utils.llm_factory import get_llm, get_embeddings
from utils.vector_store import VectorStoreManager


# =============================================================================
# 1. State 정의
# =============================================================================

class AdvancedRAGState(TypedDict):
    """
    Advanced RAG 상태
    
    Naive RAG와 달리 평가 및 수정 관련 필드가 추가됨
    """
    question: str                    # 사용자 질문
    documents: List[Document]        # 검색된 문서
    relevant_documents: List[Document]  # 관련성 있는 문서만
    context: str                     # 컨텍스트
    answer: str                      # 생성된 답변
    relevance_score: str             # 관련성 평가 ("relevant" | "not_relevant")
    hallucination_check: str         # 환각 체크 ("grounded" | "hallucinated")
    retry_count: int                 # 재시도 횟수


# =============================================================================
# 2. Vector Store 초기화
# =============================================================================

_adv_vs: VectorStoreManager = None

def get_advanced_vs() -> VectorStoreManager:
    """Advanced RAG용 Vector Store"""
    global _adv_vs
    if _adv_vs is None:
        print("📚 Advanced RAG Vector Store 초기화...")
        _adv_vs = VectorStoreManager(
            embeddings=get_embeddings(),
            collection_name="advanced_rag",
            chunk_size=400,
        )
        samples = [
            "LangGraph는 상태 기반 에이전트를 위한 프레임워크입니다. StateGraph로 노드와 엣지를 정의합니다.",
            "Self-RAG는 LLM이 검색 필요성을 스스로 판단하고, 검색 결과와 생성 응답의 품질을 평가합니다.",
            "Corrective RAG는 검색된 문서의 관련성을 평가하고, 품질이 낮으면 웹 검색으로 보완합니다.",
            "RAG 파이프라인은 검색(Retrieval), 증강(Augmentation), 생성(Generation) 3단계로 구성됩니다.",
            "Adaptive RAG는 쿼리 복잡도에 따라 단순 응답, 검색 응답, 다단계 추론을 선택합니다.",
            "Hallucination은 LLM이 사실이 아닌 정보를 생성하는 현상으로, RAG로 완화할 수 있습니다.",
        ]
        _adv_vs.add_texts(texts=samples)
        print(f"✅ {len(samples)}개 문서 로드")
    return _adv_vs


# =============================================================================
# 3. 노드 함수
# =============================================================================

def retrieve_node(state: AdvancedRAGState) -> dict:
    """문서 검색"""
    print(f"\n🔍 검색: '{state['question']}'")
    docs = get_advanced_vs().search(query=state["question"], k=4)
    print(f"   → {len(docs)}개 문서")
    return {"documents": docs}


def grade_documents_node(state: AdvancedRAGState) -> dict:
    """
    문서 관련성 평가 (Grading)
    
    LLM을 사용하여 각 문서가 질문과 관련있는지 평가합니다.
    관련 없는 문서는 필터링합니다.
    """
    print("\n📊 문서 관련성 평가...")
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """문서가 질문과 관련있으면 "yes", 없으면 "no"만 답하세요.

질문: {question}
문서: {document}

관련성 (yes/no):"""),
    ])
    
    chain = prompt | llm
    relevant_docs = []
    
    for doc in state["documents"]:
        result = chain.invoke({
            "question": state["question"],
            "document": doc.page_content[:500]
        })
        
        if "yes" in result.content.lower():
            relevant_docs.append(doc)
    
    print(f"   → 관련 문서: {len(relevant_docs)}/{len(state['documents'])}개")
    
    # 관련성 점수 결정
    score = "relevant" if len(relevant_docs) >= 2 else "not_relevant"
    
    # 컨텍스트 생성
    context = "\n\n".join([
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(relevant_docs)
    ]) if relevant_docs else ""
    
    return {
        "relevant_documents": relevant_docs,
        "relevance_score": score,
        "context": context,
    }


def generate_node(state: AdvancedRAGState) -> dict:
    """답변 생성"""
    print("\n💭 답변 생성...")
    
    if not state["context"]:
        return {"answer": "관련 정보를 찾을 수 없습니다."}
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """컨텍스트만 사용하여 답변하세요. 컨텍스트에 없는 정보는 추측하지 마세요.

컨텍스트:
{context}"""),
        ("human", "{question}"),
    ])
    
    response = (prompt | llm).invoke({
        "context": state["context"],
        "question": state["question"],
    })
    
    return {"answer": response.content}


def check_hallucination_node(state: AdvancedRAGState) -> dict:
    """
    환각 검사 (Hallucination Check)
    
    생성된 답변이 컨텍스트에 기반하는지 검증합니다.
    """
    print("\n🔬 환각 검사...")
    
    if not state["context"] or not state["answer"]:
        return {"hallucination_check": "grounded"}
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """답변이 컨텍스트에 근거하면 "grounded", 그렇지 않으면 "hallucinated"만 답하세요.

컨텍스트:
{context}

답변:
{answer}

판정 (grounded/hallucinated):"""),
    ])
    
    result = (prompt | llm).invoke({
        "context": state["context"],
        "answer": state["answer"],
    })
    
    check = "grounded" if "grounded" in result.content.lower() else "hallucinated"
    print(f"   → 결과: {check}")
    
    return {"hallucination_check": check}


def fallback_search_node(state: AdvancedRAGState) -> dict:
    """
    폴백 검색 (Fallback)
    
    관련 문서가 부족하거나 환각이 감지되면 추가 검색을 수행합니다.
    실제로는 웹 검색 등을 사용할 수 있습니다.
    """
    print("\n🔄 폴백 검색...")
    
    # 재시도 횟수 증가
    retry = state.get("retry_count", 0) + 1
    
    if retry >= 2:
        print("   → 최대 재시도 도달")
        return {
            "retry_count": retry,
            "answer": f"죄송합니다. '{state['question']}'에 대한 정확한 정보를 찾지 못했습니다."
        }
    
    # 다른 검색어로 재검색 (여기서는 단순 재검색)
    vs = get_advanced_vs()
    docs = vs.search(query=f"{state['question']} 설명", k=3)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    print(f"   → 재검색 결과: {len(docs)}개")
    return {
        "documents": docs,
        "relevant_documents": docs,
        "context": context,
        "retry_count": retry,
        "relevance_score": "relevant" if docs else "not_relevant",
    }


# =============================================================================
# 4. 라우터 함수 (조건부 분기)
# =============================================================================

def route_by_relevance(state: AdvancedRAGState) -> Literal["generate", "fallback"]:
    """관련성에 따라 분기"""
    if state.get("relevance_score") == "relevant":
        return "generate"
    return "fallback"


def route_by_hallucination(state: AdvancedRAGState) -> Literal[END, "fallback"]:
    """환각 검사 결과에 따라 분기"""
    if state.get("hallucination_check") == "grounded":
        return END
    if state.get("retry_count", 0) >= 2:
        return END
    return "fallback"


# =============================================================================
# 5. 그래프 생성
# =============================================================================

def create_advanced_rag_graph():
    """
    Advanced RAG 그래프 생성
    
    구조:
        START → retrieve → grade_documents ─┬→ generate → check_hallucination ─┬→ END
                                            │                                   │
                                            └→ fallback ←──────────────────────┘
    
    핵심 기능:
    1. 문서 관련성 평가 (Grade Documents)
    2. 관련 문서 부족 시 폴백 검색
    3. 환각 검사 (Hallucination Check)  
    4. 환각 감지 시 재검색
    """
    graph = StateGraph(AdvancedRAGState)
    
    # 노드 추가
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_hallucination", check_hallucination_node)
    graph.add_node("fallback", fallback_search_node)
    
    # 엣지
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    
    # 조건부 분기: 관련성에 따라
    graph.add_conditional_edges(
        "grade",
        route_by_relevance,
        {"generate": "generate", "fallback": "fallback"}
    )
    
    graph.add_edge("generate", "check_hallucination")
    
    # 조건부 분기: 환각에 따라
    graph.add_conditional_edges(
        "check_hallucination",
        route_by_hallucination,
        {END: END, "fallback": "fallback"}
    )
    
    # 폴백 후 재생성
    graph.add_edge("fallback", "generate")
    
    print("✅ Advanced RAG 그래프 컴파일 완료!")
    return graph.compile()


# =============================================================================
# 6. 실행
# =============================================================================

def run_advanced_rag(question: str) -> str:
    """Advanced RAG 실행"""
    graph = create_advanced_rag_graph()
    
    initial_state = {
        "question": question,
        "documents": [],
        "relevant_documents": [],
        "context": "",
        "answer": "",
        "relevance_score": "",
        "hallucination_check": "",
        "retry_count": 0,
    }
    
    print(f"\n{'='*60}\n🙋 질문: {question}\n{'='*60}")
    result = graph.invoke(initial_state)
    
    print(f"\n📊 평가 결과:")
    print(f"   - 관련성: {result['relevance_score']}")
    print(f"   - 환각 검사: {result['hallucination_check']}")
    print(f"   - 재시도: {result['retry_count']}회")
    print(f"\n🤖 답변:\n{result['answer']}\n{'='*60}")
    
    return result["answer"]


def visualize_graph():
    """그래프 구조 시각화"""
    print("\n📊 Advanced RAG 그래프 (Mermaid)")
    print("```mermaid")
    print("graph TD")
    print("    START --> retrieve[검색]")
    print("    retrieve --> grade[관련성 평가]")
    print("    grade -->|relevant| generate[생성]")
    print("    grade -->|not_relevant| fallback[폴백 검색]")
    print("    generate --> check[환각 검사]")
    print("    check -->|grounded| END")
    print("    check -->|hallucinated| fallback")
    print("    fallback --> generate")
    print("```")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Advanced RAG 예제 - Self-RAG & Corrective RAG")
    print("="*60)
    
    if not get_settings().validate_openai_key():
        print("\n⚠️ OPENAI_API_KEY를 설정해주세요.")
        sys.exit(1)
    
    visualize_graph()
    
    queries = [
        "Self-RAG란 무엇인가요?",
        "Hallucination을 방지하는 방법은?",
        "파이썬으로 웹서버 만드는 법은?",  # 관련 없는 질문 테스트
    ]
    
    for q in queries:
        try:
            run_advanced_rag(q)
        except Exception as e:
            print(f"❌ 오류: {e}")
        print()
