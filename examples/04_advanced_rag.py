# -*- coding: utf-8 -*-
"""
============================================================================
📚 04. Advanced RAG 예제 - Self-RAG & Corrective RAG
============================================================================

이 예제는 검색 품질과 답변 정확성을 높이기 위한 고급 RAG 패턴을 구현합니다.
Self-RAG의 개념을 도입하여, 검색된 문서의 관련성을 평가하고(Grading),
답변이 환각(Hallucination)인지 검사하며, 필요 시 재검색(Fallback)을 수행합니다.

🎯 학습 목표:
    1. 문서 관련성 평가(Relevance Grading) 노드 구현
    2. 조건부 엣지(Conditional Edge)를 이용한 흐름 제어 및 루프
    3. 환각 감지 및 수정 전략 (Corrective RAG)

💡 핵심 개념:
    - Self-RAG: LLM이 스스로 검색 결과와 생성 품질을 평가
    - Corrective RAG: 관련 없는 검색 결과 시 쿼리 재작성/웹 검색
    - Grading: LLM을 이용해 문서-질문 관련성 점수 부여
    - Hallucination: LLM이 사실이 아닌 정보를 생성하는 현상

그래프 구조:
                          ┌────────────────────┐
                          ↓                    │
    START → retrieve → grade_documents ─┬→ generate → END
                                        │
                                        └→ rewrite_query ─┘
                                        (관련 없으면 루프)

실행 방법:
    python examples/04_advanced_rag.py
    
    실행 후 CLI에서 질문을 입력하면 Advanced RAG Agent가 응답합니다.
    종료: 'quit', 'exit', 또는 'q' 입력
"""

# =============================================================================
# 📦 필수 라이브러리 임포트
# =============================================================================

# Python 표준 라이브러리
import sys                              # 시스템 경로 조작용
import os                               # 환경변수 접근용
from pathlib import Path                # 파일 경로 처리
from typing import TypedDict, List, Literal  
# Literal: 특정 값만 허용하는 타입 (예: Literal["yes", "no"])

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
# Document: 검색된 문서 객체

from langchain_core.prompts import ChatPromptTemplate
# ChatPromptTemplate: 평가/변환용 프롬프트 템플릿

from langchain_core.output_parsers import JsonOutputParser
# JsonOutputParser: 구조화된 평가 결과 파싱

# -----------------------------------------------------------------------------
# 🔗 LangGraph 핵심 모듈 임포트
# -----------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END
# StateGraph: 그래프 빌더
# START/END: 시작점/종료점

# -----------------------------------------------------------------------------
# 🔗 프로젝트 내부 유틸리티 임포트
# -----------------------------------------------------------------------------

from utils.llm_factory import get_embeddings, log_llm_error
from utils.vector_store import VectorStoreManager


# =============================================================================
# 📋 1. State 정의
# =============================================================================
#
# Advanced RAG에서는 평가 결과와 루프 제어를 위한 필드가 추가됩니다.
# =============================================================================

class AdvancedRAGState(TypedDict):
    """
    Advanced RAG 상태
    
    평가 결과(grading)와 재시도 횟수(loop control)를 상태로 관리합니다.
    
    필드 설명:
    - question: 사용자 질문 (루프 중 재작성될 수 있음)
    - documents: 검색된 문서 리스트
    - answer: 최종 답변
    - grade: 문서 관련성 평가 결과 ("relevant" 또는 "irrelevant")
    - hallucination: 환각 여부 ("yes" 또는 "no")
    - loop_count: 재시도 횟수 (무한 루프 방지용 카운터)
    """
    question: str                # 사용자 질문
    documents: List[Document]    # 검색된 문서들
    answer: str                  # 최종 답변
    grade: str                   # "relevant" or "irrelevant"
    hallucination: str           # "yes" or "no"
    loop_count: int              # 무한 루프 방지 카운터


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

def retrieve(state: AdvancedRAGState):
    """
    문서 검색 노드
    
    현재 질문을 기반으로 관련 문서를 검색합니다.
    질문이 재작성되면 새로운 질문으로 다시 검색됩니다.
    """
    print(f"\n🔍 검색 수행: {state['question']}")
    
    vs = get_vector_store()
    docs = vs.search(state["question"], k=3)
    
    print(f"   → {len(docs)}개 문서 검색됨")
    for i, doc in enumerate(docs):
        print(f"   [{i+1}] {doc.page_content[:50]}...")
    
    return {"documents": docs}


def grade_documents(state: AdvancedRAGState):
    """
    문서 관련성 평가 노드 (Grading)
    
    LLM을 사용하여 검색된 문서가 질문과 관련이 있는지 평가합니다.
    
    💡 Grading 전략:
       - 검색된 문서 각각에 대해 관련성 평가
       - 하나라도 관련 있으면 "relevant"
       - 모두 관련 없으면 "irrelevant" → 재검색 트리거
    """
    print("📊 문서 평가 중...")
    
    # AI 모델 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    # 평가용 프롬프트
    prompt = ChatPromptTemplate.from_template(
        """당신은 문서 평가자입니다. 다음 문서가 사용자의 질문과 관련이 있는지 평가하세요.
        관련이 있다면 'yes', 없다면 'no'라고만 답하세요.

        질문: {question}
        문서: {document}
        """
    )
    
    chain = prompt | model
    
    # 각 문서를 평가하여 하나라도 관련 있으면 relevant
    is_relevant = False
    for i, doc in enumerate(state["documents"]):
        res = chain.invoke({
            "question": state["question"], 
            "document": doc.page_content
        })
        
        # 응답에서 yes/no 판단
        if "yes" in res.content.lower():
            print(f"   → 문서 {i+1}: 관련 있음 ✓")
            is_relevant = True
            break  # 하나라도 관련 있으면 충분
        else:
            print(f"   → 문서 {i+1}: 관련 없음 ✗")

    grade = "relevant" if is_relevant else "irrelevant"
    print(f"   📋 최종 평가: {grade}")
    
    return {"grade": grade}


def generate(state: AdvancedRAGState):
    """
    답변 생성 노드
    
    검색된 문서를 컨텍스트로 사용하여 최종 답변을 생성합니다.
    """
    print("📝 답변 생성 중...")
    
    # 컨텍스트 구성
    context = "\n".join(d.page_content for d in state["documents"])
    
    # AI 모델 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    res = model.invoke(f"컨텍스트: {context}\n\n질문: {state['question']}\n답변:")
    
    return {"answer": res.content}


def rewrite_query(state: AdvancedRAGState):
    """
    질문 재작성 노드 (Fallback)
    
    검색된 문서가 관련 없을 때 질문을 개선합니다.
    
    💡 재작성 전략:
       - 실제로는 LLM을 이용해 더 나은 쿼리 생성
       - 여기서는 단순히 "(상세 설명)" 추가로 시뮬레이션
       - loop_count 증가로 무한 루프 방지
    """
    print("🔄 질문 재작성 중...")
    
    # 현재 재시도 횟수
    current_count = state.get("loop_count", 0)
    
    # 질문 개선 (실제로는 LLM 활용)
    # 예: "Self-RAG가 뭐야?" → "Self-RAG가 뭐야? (상세 설명)"
    new_query = state["question"] + " (상세 설명)"
    
    print(f"   → 기존: {state['question']}")
    print(f"   → 변경: {new_query}")
    print(f"   → 재시도 횟수: {current_count + 1}")
    
    return {
        "question": new_query,
        "loop_count": current_count + 1
    }


# =============================================================================
# 🚦 4. 조건부 엣지 함수
# =============================================================================
#
# 조건부 엣지는 현재 상태를 보고 "다음에 어떤 노드로 갈지" 결정합니다.
# 반환값은 노드 이름 문자열이어야 합니다.
# =============================================================================

def check_relevance(state: AdvancedRAGState) -> Literal["generate", "rewrite_query", "end"]:
    """
    평가 결과에 따른 분기 처리
    
    Returns:
        "generate": 관련 문서 있음 → 답변 생성으로
        "rewrite_query": 관련 문서 없음 → 질문 재작성으로
        "end": 최대 재시도 초과 → 종료
        
    💡 Literal 타입:
       반환값이 정해진 문자열 중 하나임을 명시
       IDE에서 자동완성과 타입 체크 지원
    """
    # 무한 루프 방지: 최대 2회 재시도
    if state.get("loop_count", 0) > 1:
        print("   ⚠️ 최대 재시도 횟수 초과, 강제 종료")
        return "end"

    if state["grade"] == "relevant":
        print("   ✅ 관련 문서 확인됨 → 답변 생성으로 이동")
        return "generate"
    else:
        print("   🔄 관련 문서 없음 → 질문 재작성으로 이동")
        return "rewrite_query"


# =============================================================================
# 🔀 5. 그래프 구성
# =============================================================================

def create_graph():
    """
    Advanced RAG 그래프 생성
    
    그래프 구조:
    
                              ┌────────────────────┐
                              ↓                    │
        START → retrieve → grade_documents ─┬→ generate → END
                                            │
                                            ├→ rewrite_query ─┘
                                            │
                                            └→ END (최대 재시도 시)
                                            
    💡 루프 구조:
       - 조건부 엣지에서 "rewrite_query" 선택 시
       - rewrite_query → retrieve로 다시 돌아감
       - 이것이 LangGraph의 순환(Cycle) 기능
    """
    builder = StateGraph(AdvancedRAGState)
    
    # -------------------------------------------------------------------------
    # 노드 추가
    # -------------------------------------------------------------------------
    
    builder.add_node("retrieve", retrieve)              # 문서 검색
    builder.add_node("grade_documents", grade_documents)  # 관련성 평가
    builder.add_node("generate", generate)              # 답변 생성
    builder.add_node("rewrite_query", rewrite_query)    # 질문 재작성
    
    # -------------------------------------------------------------------------
    # 엣지 연결
    # -------------------------------------------------------------------------
    
    # 시작 → 검색
    builder.add_edge(START, "retrieve")
    
    # 검색 → 평가
    builder.add_edge("retrieve", "grade_documents")
    
    # 조건부 엣지: 평가 결과에 따라 분기
    # add_conditional_edges(source, condition_func, path_map)
    # - source: 조건을 평가할 노드
    # - condition_func: 상태를 받아 다음 노드 이름을 반환하는 함수
    # - path_map: 반환값 → 실제 노드 매핑 (선택사항)
    builder.add_conditional_edges(
        "grade_documents",       # 이 노드가 끝나면
        check_relevance,         # 이 함수를 호출하여
        {                        # 반환값에 따라 다음 노드 결정
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "end": END           # END로 직접 갈 수도 있음
        }
    )
    
    # 루프: 재작성 후 다시 검색
    # 이 엣지가 순환(Cycle)을 만듦!
    builder.add_edge("rewrite_query", "retrieve")
    
    # 답변 생성 후 종료
    builder.add_edge("generate", END)
    
    return builder.compile()


# =============================================================================
# ▶️ 6. 실행 함수
# =============================================================================

def run_advanced_rag(question: str):
    """
    Advanced RAG 파이프라인을 실행하여 질문에 답변합니다.
    """
    app = create_graph()
    
    print(f"\n{'='*60}")
    print(f"🙋 질문: {question}")
    print('='*60)
    
    try:
        # 초기 상태: 질문과 loop_count 설정
        result = app.invoke({
            "question": question, 
            "loop_count": 0
        })
        
        if result.get("answer"):
            print(f"\n🤖 답변: {result['answer']}")
        else:
            print("\n🤖 관련 정보를 찾지 못해 답변을 생성하지 못했습니다.")
            
    except Exception as e:
        log_llm_error(e)
        print("❌ 실행 중 오류가 발생했습니다.")


# =============================================================================
# 🚀 7. 메인 실행부 (CLI 인터페이스)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📚 LangGraph Advanced RAG Example (Self-Correction)")
    print("="*60)
    print("CLI 모드로 실행됩니다. 질문을 입력하세요.")
    print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.\n")
    
    while True:
        try:
            question = input("🙋 질문을 입력하세요: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ("quit", "exit", "q"):
                print("👋 Advanced RAG Agent를 종료합니다. 안녕히 가세요!")
                break
            
            run_advanced_rag(question)
            
        except KeyboardInterrupt:
            print("\n👋 Advanced RAG Agent를 종료합니다. (Ctrl+C)")
            break
        except EOFError:
            print("\n👋 Advanced RAG Agent를 종료합니다. (EOF)")
            break
