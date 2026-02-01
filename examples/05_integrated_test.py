# -*- coding: utf-8 -*-
"""
============================================================================
📚 05. Integrated RAG - Entity + Advanced + Adaptive 통합 시스템
============================================================================

03_entity_rag, 04_advanced_rag, 04a_adaptive_rag의 기법을 하나로 통합한 
최종 완성형 RAG Agent입니다.

🎯 통합된 핵심 기술:
    1. Adaptive Router: 질문 난이도(simple/moderate/complex) 자동 판별
    2. Entity RAG: 엔티티 추출 및 병렬 검색 (Fan-out/Fan-in)
    3. Advanced RAG: 문서 평가(Grading) 및 쿼리 재작성 루프
    4. 공통 데이터 로더: Vector Store 영속화 및 파일 변경 감지

그래프 구조:
                                 ┌→ direct_answer ──────────────────────→ END (simple)
                                 │
    START → classify ────────────┼→ entity_search ─┬→ semantic_search ─┐
                                 │                  │                   │
                                 │                  └→ merge ───────────┘
                                 │                           │
                                 │                           ↓
                                 └→ complex_rag ─┬→ grade_docs ─┬→ generate → END
                                                 │              │
                                                 │              └→ rewrite → retrieve ─┘
                                                 │
                                                 └→ multi_step_rag → END

실행 방법:
    python examples/05_integrated_test.py
"""

import sys
import os
from pathlib import Path
from typing import TypedDict, List, Literal

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# LangChain 구성 요소
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# LangGraph 구성 요소
from langgraph.graph import StateGraph, START, END

# 프로젝트 유틸리티
from utils.llm_factory import log_llm_error
from utils.vector_store import VectorStoreManager
from utils.data_loader import get_rag_vector_store


# =============================================================================
# 📋 1. 통합 상태(State) 정의
# =============================================================================

class IntegratedRAGState(TypedDict):
    """
    통합 RAG 시스템의 상태
    
    모든 RAG 기법에서 필요한 필드들을 통합합니다.
    """
    # 기본 필드
    question: str                    # 사용자 질문
    answer: str                      # 최종 답변
    
    # Adaptive RAG (04a) 필드
    query_complexity: str            # 질문 난이도 (simple/moderate/complex)
    strategy_used: str               # 사용된 전략 이름
    
    # Entity RAG (03) 필드
    entities: List[str]              # 추출된 엔티티 리스트
    entity_docs: List[Document]      # 엔티티 기반 검색 결과
    semantic_docs: List[Document]    # 의미론적 검색 결과
    
    # Advanced RAG (04) 필드
    documents: List[Document]        # 병합/검색된 문서 리스트
    grade: str                       # 문서 관련성 평가 (relevant/irrelevant)
    loop_count: int                  # 쿼리 재작성 루프 카운터
    
    # 디버깅용
    steps_taken: List[str]           # 거쳐온 노드 기록


# =============================================================================
# 🗄️ 2. Vector Store 초기화 (공통 모듈 사용)
# =============================================================================

def get_vector_store() -> VectorStoreManager:
    """
    통합 RAG용 Vector Store 초기화
    
    모든 기능이 같은 collection을 공유하여 임베딩을 재사용합니다.
    """
    return get_rag_vector_store(collection_name="integrated_rag")


# =============================================================================
# 🧠 3. Adaptive RAG 노드: 질문 분류 (04a 기법)
# =============================================================================

def classify_query(state: IntegratedRAGState) -> dict:
    """
    [Adaptive] 질문 난이도를 분류합니다.
    
    - simple: 검색 없이 바로 답변 가능한 간단한 질문
    - moderate: 일반적인 RAG 검색이 필요한 질문
    - complex: 엔티티 추출 + 다단계 분석이 필요한 복잡한 질문
    """
    print(f"\n🧐 [분류] 질문 난이도 분석 중...")
    
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """질문을 분석하여 3가지 중 하나로 분류하세요. 단어 하나만 답하세요.
1. "simple": 인사, 시간 묻기, 상식적인 질문
2. "moderate": 한 번의 검색으로 답변 가능한 일반 질문  
3. "complex": 여러 개념 비교, 심층 분석이 필요한 복잡한 질문"""),
        ("human", "{question}"),
    ])
    
    response = (prompt | model).invoke({"question": state["question"]})
    complexity = response.content.lower().strip()
    
    # 유효하지 않은 응답은 moderate로 기본 설정
    if complexity not in ["simple", "moderate", "complex"]:
        complexity = "moderate"
    
    print(f"   → 판단 결과: '{complexity}' 수준")
    
    return {
        "query_complexity": complexity,
        "steps_taken": ["classify"]
    }


# =============================================================================
# ⚡ 4. Simple 전략: 직접 답변 (04a 기법)
# =============================================================================

def direct_answer(state: IntegratedRAGState) -> dict:
    """
    [Simple] 검색 없이 LLM의 지식으로 직접 답변합니다.
    """
    print("⚡ [Simple] 검색 없이 바로 답변합니다.")
    
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    response = model.invoke(state["question"])
    
    return {
        "answer": response.content,
        "strategy_used": "Simple (직접 답변)",
        "steps_taken": state["steps_taken"] + ["direct_answer"]
    }


# =============================================================================
# 🏷️ 5. Entity RAG 노드들 (03 기법)
# =============================================================================

def extract_entities(state: IntegratedRAGState) -> dict:
    """
    [Entity RAG] LLM을 사용하여 질문에서 핵심 엔티티를 추출합니다.
    """
    print("🏷️ [Entity] 엔티티 추출 중...")
    
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """질문에서 핵심 키워드(엔티티)를 추출하여 JSON 리스트로 반환하세요.
예시: {{"entities": ["LangGraph", "RAG"]}}
질문: {question}"""),
    ])
    
    try:
        chain = prompt | model | JsonOutputParser()
        result = chain.invoke({"question": state["question"]})
        entities = result.get("entities", [])
        print(f"   → 추출된 엔티티: {entities}")
        return {"entities": entities}
    except Exception as e:
        print(f"   → 추출 실패, 빈 리스트 반환: {e}")
        return {"entities": []}


def search_by_entity(state: IntegratedRAGState) -> dict:
    """
    [Entity RAG] 엔티티 기반 검색 (병렬 실행 1)
    """
    print("🔍 [Entity] 엔티티 기반 검색 수행...")
    
    vs = get_vector_store()
    results = []
    
    for entity in state.get("entities", []):
        docs = vs.search(entity, k=1)
        results.extend(docs)
        print(f"   → '{entity}' 검색: {len(docs)}개 문서")
    
    return {"entity_docs": results}


def search_semantic(state: IntegratedRAGState) -> dict:
    """
    [Entity RAG] 의미론적 검색 (병렬 실행 2)
    """
    print("🔍 [Semantic] 의미론적 검색 수행...")
    
    vs = get_vector_store()
    docs = vs.search(state["question"], k=2)
    print(f"   → {len(docs)}개 문서 검색됨")
    
    return {"semantic_docs": docs}


def merge_results(state: IntegratedRAGState) -> dict:
    """
    [Entity RAG] 엔티티 + 의미론적 검색 결과 병합 (Fan-in)
    """
    print("🔄 [Merge] 검색 결과 병합 중...")
    
    seen = set()
    merged = []
    
    all_docs = state.get("entity_docs", []) + state.get("semantic_docs", [])
    
    for doc in all_docs:
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    
    print(f"   → 총 {len(merged)}개 문서 병합됨")
    
    return {
        "documents": merged,
        "steps_taken": state["steps_taken"] + ["entity_search", "semantic_search", "merge"]
    }


# =============================================================================
# 📊 6. Advanced RAG 노드들 (04 기법)
# =============================================================================

def grade_documents(state: IntegratedRAGState) -> dict:
    """
    [Advanced] 검색된 문서의 관련성을 평가합니다 (Grading)
    """
    print("📊 [Grade] 문서 관련성 평가 중...")
    
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    prompt = ChatPromptTemplate.from_template(
        """문서가 질문과 관련이 있으면 'yes', 없으면 'no'라고만 하세요.
질문: {question}
문서: {document}"""
    )
    
    chain = prompt | model
    
    is_relevant = False
    for i, doc in enumerate(state.get("documents", [])):
        res = chain.invoke({
            "question": state["question"],
            "document": doc.page_content
        })
        
        if "yes" in res.content.lower():
            print(f"   → 문서 {i+1}: 관련 있음 ✓")
            is_relevant = True
            break
        else:
            print(f"   → 문서 {i+1}: 관련 없음 ✗")
    
    grade = "relevant" if is_relevant else "irrelevant"
    print(f"   📋 최종 평가: {grade}")
    
    return {
        "grade": grade,
        "steps_taken": state["steps_taken"] + ["grade_documents"]
    }


def rewrite_query(state: IntegratedRAGState) -> dict:
    """
    [Advanced] 관련 문서가 없을 때 질문을 재작성합니다 (Fallback)
    """
    print("🔄 [Rewrite] 질문 재작성 중...")
    
    current_count = state.get("loop_count", 0)
    
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    # LLM을 사용하여 더 나은 검색 쿼리 생성
    response = model.invoke(
        f"다음 질문을 검색에 더 적합하게 다시 작성해주세요. 질문만 출력하세요.\n원본: {state['question']}"
    )
    new_query = response.content.strip()
    
    print(f"   → 기존: {state['question']}")
    print(f"   → 변경: {new_query}")
    print(f"   → 재시도 횟수: {current_count + 1}")
    
    return {
        "question": new_query,
        "loop_count": current_count + 1,
        "steps_taken": state["steps_taken"] + ["rewrite_query"]
    }


def retrieve_for_rewrite(state: IntegratedRAGState) -> dict:
    """
    [Advanced] 재작성된 질문으로 다시 검색합니다
    """
    print(f"🔍 [Retrieve] 재검색 수행: {state['question']}")
    
    vs = get_vector_store()
    docs = vs.search(state["question"], k=3)
    
    print(f"   → {len(docs)}개 문서 검색됨")
    
    return {"documents": docs}


# =============================================================================
# 📝 7. 답변 생성 노드
# =============================================================================

def generate_answer(state: IntegratedRAGState) -> dict:
    """
    검색된 문서를 기반으로 최종 답변을 생성합니다.
    """
    print("📝 [Generate] 답변 생성 중...")
    
    context = "\n".join(d.page_content for d in state.get("documents", []))
    
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    response = model.invoke(f"참고 문서:\n{context}\n\n질문: {state['question']}\n\n답변:")
    
    return {
        "answer": response.content,
        "strategy_used": "Advanced RAG (Entity + Grading)",
        "steps_taken": state["steps_taken"] + ["generate"]
    }


def generate_fallback_answer(state: IntegratedRAGState) -> dict:
    """
    관련 문서를 찾지 못했을 때 LLM 지식으로 답변합니다.
    """
    print("📝 [Fallback] 관련 문서 없음, 일반 지식으로 답변...")
    
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    response = model.invoke(state["question"])
    
    return {
        "answer": response.content,
        "strategy_used": "Fallback (일반 LLM)",
        "steps_taken": state["steps_taken"] + ["fallback_generate"]
    }


# =============================================================================
# 🔬 8. Complex 전략: 다단계 분석 (04a 기법)
# =============================================================================

def complex_multi_step_rag(state: IntegratedRAGState) -> dict:
    """
    [Complex] 질문을 분해하여 다단계로 분석합니다.
    """
    print("🔬 [Complex] 다단계 정밀 분석 수행...")
    
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    # 1. 질문 분해
    decompose_res = model.invoke(
        f"이 질문을 해결하기 위해 먼저 알아야 할 세부 질문 2개를 작성하세요. 한 줄씩 쓰세요.\n질문: {state['question']}"
    )
    sub_queries = [q.strip() for q in decompose_res.content.split("\n") if q.strip()][:2]
    print(f"   → 세부 질문: {sub_queries}")
    
    # 2. 각 세부 질문 + 원본 질문으로 검색
    vs = get_vector_store()
    all_context = []
    
    for sq in sub_queries + [state["question"]]:
        docs = vs.search(sq, k=2)
        all_context.extend([d.page_content for d in docs])
    
    # 3. 중복 제거 및 심층 답변 생성
    final_context = "\n".join(list(set(all_context)))
    
    response = model.invoke(
        f"다음 정보를 바탕으로 심층 분석 답변을 작성하세요.\n\n참고 정보:\n{final_context}\n\n질문: {state['question']}"
    )
    
    return {
        "answer": response.content,
        "strategy_used": "Complex (다단계 정밀 RAG)",
        "steps_taken": state["steps_taken"] + ["complex_multi_step"]
    }


# =============================================================================
# 🚦 9. 조건부 라우팅 함수들
# =============================================================================

def route_by_complexity(state: IntegratedRAGState) -> Literal["simple", "moderate", "complex"]:
    """
    분류된 난이도에 따라 경로를 결정합니다.
    """
    return state["query_complexity"]


def check_grade_and_loop(state: IntegratedRAGState) -> Literal["generate", "rewrite", "fallback"]:
    """
    문서 평가 결과와 루프 횟수에 따라 다음 단계를 결정합니다.
    """
    # 최대 재시도 횟수 초과 시 fallback
    if state.get("loop_count", 0) > 1:
        print("   ⚠️ 최대 재시도 횟수 초과 → fallback")
        return "fallback"
    
    if state.get("grade") == "relevant":
        print("   ✅ 관련 문서 확인됨 → 답변 생성")
        return "generate"
    else:
        print("   🔄 관련 문서 없음 → 질문 재작성")
        return "rewrite"


# =============================================================================
# 🔗 10. 그래프 조립
# =============================================================================

def create_graph():
    """
    모든 RAG 기법을 통합한 그래프를 생성합니다.
    """
    builder = StateGraph(IntegratedRAGState)
    
    # -------------------------------------------------------------------------
    # 노드 등록
    # -------------------------------------------------------------------------
    
    # Adaptive 분류
    builder.add_node("classify", classify_query)
    
    # Simple 전략
    builder.add_node("direct_answer", direct_answer)
    
    # Moderate 전략 (Entity RAG)
    builder.add_node("extract_entities", extract_entities)
    builder.add_node("entity_search", search_by_entity)
    builder.add_node("semantic_search", search_semantic)
    builder.add_node("merge", merge_results)
    
    # Advanced RAG (Grading + Rewrite)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve", retrieve_for_rewrite)
    builder.add_node("generate", generate_answer)
    builder.add_node("fallback_generate", generate_fallback_answer)
    
    # Complex 전략
    builder.add_node("complex_rag", complex_multi_step_rag)
    
    # -------------------------------------------------------------------------
    # 엣지 연결
    # -------------------------------------------------------------------------
    
    # 시작 → 분류
    builder.add_edge(START, "classify")
    
    # 난이도별 분기
    builder.add_conditional_edges(
        "classify",
        route_by_complexity,
        {
            "simple": "direct_answer",
            "moderate": "extract_entities",
            "complex": "complex_rag"
        }
    )
    
    # Simple 종료
    builder.add_edge("direct_answer", END)
    
    # Complex 종료
    builder.add_edge("complex_rag", END)
    
    # Moderate: Entity RAG 병렬 검색
    builder.add_edge("extract_entities", "entity_search")
    builder.add_edge("extract_entities", "semantic_search")
    builder.add_edge("entity_search", "merge")
    builder.add_edge("semantic_search", "merge")
    
    # Moderate: Advanced RAG (Grading + Rewrite 루프)
    builder.add_edge("merge", "grade_documents")
    
    builder.add_conditional_edges(
        "grade_documents",
        check_grade_and_loop,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
            "fallback": "fallback_generate"
        }
    )
    
    # Rewrite 루프
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    
    # 생성 종료
    builder.add_edge("generate", END)
    builder.add_edge("fallback_generate", END)
    
    return builder.compile()


# =============================================================================
# ▶️ 11. 실행 함수 및 CLI
# =============================================================================

def run_integrated_rag(question: str, app):
    """
    통합 RAG 파이프라인을 실행합니다.
    """
    print(f"\n{'='*60}")
    print(f"🙋 질문: {question}")
    print("="*60)
    
    try:
        result = app.invoke({
            "question": question,
            "query_complexity": "",
            "strategy_used": "",
            "entities": [],
            "entity_docs": [],
            "semantic_docs": [],
            "documents": [],
            "grade": "",
            "loop_count": 0,
            "answer": "",
            "steps_taken": []
        })
        
        print(f"\n📊 사용된 전략: {result.get('strategy_used', 'Unknown')}")
        print(f"💡 실행 경로: {' → '.join(result.get('steps_taken', []))}")
        print(f"\n🤖 답변:\n{result.get('answer', '답변 생성 실패')}")
        
    except Exception as e:
        log_llm_error(e)
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 통합 RAG 시스템 (Entity + Advanced + Adaptive)")
    print("="*60)
    print("- 질문 난이도에 따라 최적의 RAG 전략을 자동 선택합니다.")
    print("- 종료: 'quit', 'exit', 또는 'q'")
    print("="*60)
    
    # 그래프 생성
    app = create_graph()
    
    while True:
        try:
            user_input = input("\n🙋 질문: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 통합 RAG Agent를 종료합니다. 안녕히 가세요!")
                break
            
            run_integrated_rag(user_input, app)
            
        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"\n⚠️ 오류 발생: {e}")
