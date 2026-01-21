# -*- coding: utf-8 -*-
"""
04. Advanced RAG 예제 - Self-RAG & Corrective RAG

이 예제는 검색 품질과 답변 정확성을 높이기 위한 고급 RAG 패턴을 구현합니다.
Self-RAG의 개념을 도입하여, 검색된 문서의 관련성을 평가하고(Grading),
답변이 환각(Hallucination)인지 검사하며, 필요 시 재검색(Fallback)을 수행합니다.

학습 목표:
    1. 문서 관련성 평가(Relevance Grading) 노드 구현
    2. 조건부 엣지(Conditional Edge)를 이용한 흐름 제어 및 루프
    3. 환각 감지 및 수정 전략 (Corrective RAG)

실행 방법:
    python examples/04_advanced_rag.py
"""

import sys
from pathlib import Path
from typing import TypedDict, List, Literal

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END

from config.settings import get_settings
from utils.llm_factory import get_llm, get_embeddings, log_llm_error
from utils.vector_store import VectorStoreManager


# =============================================================================
# 1. State 정의
# =============================================================================

class AdvancedRAGState(TypedDict):
    """
    Advanced RAG 상태
    
    평가 결과(grading)와 재시도 횟수(loop control)를 상태로 관리합니다.
    """
    question: str
    documents: List[Document]
    answer: str
    grade: str               # "relevant" or "irrelevant"
    hallucination: str       # "yes" or "no"
    loop_count: int          # 무한 루프 방지 카운터


# =============================================================================
# 2. Vector Store 준비
# =============================================================================

def get_vector_store() -> VectorStoreManager:
    embeddings = get_embeddings()
    manager = VectorStoreManager(embeddings=embeddings, collection_name="advanced_rag")

    if True:
        texts = [
            "Self-RAG는 LLM이 스스로 검색 필요성을 판단하고 생성된 답변을 비평(Critique)하는 프레임워크입니다.",
            "Corrective RAG(CRAG)는 검색된 문서가 질문과 관련이 없는 경우 웹 검색 등을 통해 지식을 수정/보완합니다.",
            "LangGraph는 순환(Cycle)이 있는 그래프를 통해 에이전트의 자기 수정(Self-Correction) 패턴을 지원합니다.",
            "Hallucination(환각)은 LLM이 사실이 아닌 정보를 그럴듯하게 생성하는 현상입니다.",
        ]
        manager.add_texts(texts)

    return manager


# =============================================================================
# 3. 노드 함수 정의
# =============================================================================

def retrieve(state: AdvancedRAGState):
    """문서 검색 노드"""
    print(f"\n🔍 검색 수행: {state['question']}")
    vs = get_vector_store()
    docs = vs.search(state["question"], k=3)
    return {"documents": docs}


def grade_documents(state: AdvancedRAGState):
    """문서 관련성 평가 노드 (Grading)"""
    print("📊 문서 평가 중...")
    
    llm = get_llm()
    # Pydantic OutputParser를 쓰면 더 좋지만, 여기선 프롬프트로 간단히 처리
    prompt = ChatPromptTemplate.from_template(
        """당신은 문서 평가자입니다. 다음 문서가 사용자의 질문과 관련이 있는지 평가하세요.
        관련이 있다면 'yes', 없다면 'no'라고만 답하세요.

        질문: {question}
        문서: {document}
        """
    )
    
    chain = prompt | llm
    
    # 간소화를 위해 첫 번째 문서만 평가하거나, 전체를 평가해서 하나라도 관련 있으면 pass 등 전략 선택 가능
    # 여기서는 검색된 문서 중 하나라도 관련 있으면 'relevant'로 판단
    is_relevant = False
    for doc in state["documents"]:
        res = chain.invoke({"question": state["question"], "document": doc.page_content})
        if "yes" in res.content.lower():
            is_relevant = True
            break

    grade = "relevant" if is_relevant else "irrelevant"
    print(f"   -> 평가 결과: {grade}")
    
    return {"grade": grade}


def generate(state: AdvancedRAGState):
    """답변 생성 노드"""
    print("📝 답변 생성 중...")
    context = "\n".join(d.page_content for d in state["documents"])
    
    llm = get_llm()
    res = llm.invoke(f"컨텍스트: {context}\n\n질문: {state['question']}\n답변:")
    
    return {"answer": res.content}


def rewrite_query(state: AdvancedRAGState):
    """질문 재작성 노드 (Fallback)"""
    print("🔄 질문 재작성 중...")
    
    # 실제로는 LLM을 이용해 쿼리를 개선하겠지만, 여기선 단순히 뒤에 '설명'을 붙이는 예시
    new_query = state["question"] + " (상세 설명)"
    
    return {
        "question": new_query,
        "loop_count": state.get("loop_count", 0) + 1
    }


# =============================================================================
# 4. 조건부 엣지 함수
# =============================================================================

def check_relevance(state: AdvancedRAGState) -> Literal["generate", "rewrite_query", "end"]:
    """평가 결과에 따른 분기 처리"""

    # 무한 루프 방지 (최대 2회 재시도)
    if state.get("loop_count", 0) > 1:
        print("   -> 최대 재시도 횟수 초과, 종료")
        return "end"

    if state["grade"] == "relevant":
        print("   -> 관련 문서 확인됨, 답변 생성으로 이동")
        return "generate"
    else:
        print("   -> 관련 문서 없음, 질문 재작성으로 이동")
        return "rewrite_query"


# =============================================================================
# 5. 그래프 구성
# =============================================================================

def create_advanced_rag_graph():
    builder = StateGraph(AdvancedRAGState)
    
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("generate", generate)
    builder.add_node("rewrite_query", rewrite_query)
    
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    
    # 조건부 엣지
    builder.add_conditional_edges(
        "grade_documents",
        check_relevance,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "end": END
        }
    )
    
    builder.add_edge("rewrite_query", "retrieve") # 루프: 재작성 후 다시 검색
    builder.add_edge("generate", END)
    
    return builder.compile()


# =============================================================================
# 6. 실행 및 테스트
# =============================================================================

if __name__ == "__main__":
    print("\nLangGraph Advanced RAG Example (Self-Correction)")
    
    graph = create_advanced_rag_graph()
    
    # 1. 정상 질문
    q1 = "Self-RAG가 뭐야?"
    # 2. 관련 없는 질문 (재작성 유도용)
    q2 = "오늘 점심 메뉴 추천해줘"
    
    for q in [q1, q2]:
        print(f"\n{'='*40}\n질문: {q}\n{'='*40}")
        try:
            # 초기 상태에 loop_count 0 설정
            result = graph.invoke({"question": q, "loop_count": 0})
            if result.get("answer"):
                print(f"\n🤖 답변: {result['answer']}")
            else:
                print("\n🤖 답변을 생성하지 못했습니다.")
        except Exception as e:
            log_llm_error(e)
