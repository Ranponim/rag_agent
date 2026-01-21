# -*- coding: utf-8 -*-
"""
03. Entity RAG 예제 - 엔티티 기반 병렬 검색

LangGraph의 병렬 실행(Parallel Execution) 기능을 활용하여,
엔티티 기반 검색과 의미론적 검색을 동시에 수행하고 결과를 병합하는 패턴을 학습합니다.

학습 목표:
    1. LangGraph의 병렬 노드 실행 (Fan-out / Fan-in) 패턴 구현
    2. LLM을 이용한 엔티티 추출 (Structured Output)
    3. 다중 검색 결과 병합 (Merge) 전략

실행 방법:
    python examples/03_entity_rag.py
"""

import sys
from pathlib import Path
from typing import TypedDict, List

# 프로젝트 루트를 경로에 추가하여 내부 모듈(config, utils)을 불러올 수 있게 함
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangChain: 엔티티 추출 및 문서 검색 관련
from langchain_core.documents import Document  # 검색된 데이터의 표준 문서 객체
from langchain_core.prompts import ChatPromptTemplate  # 쿼리 분석용 프롬프트 설계도
from langchain_core.output_parsers import JsonOutputParser  # 추출된 엔티티를 파이썬 리스트로 변환
from langgraph.graph import StateGraph, START, END  # 병렬 실행 흐름 제어를 위한 그래프 구성 도구

# 프로젝트 유틸리티
from config.settings import get_settings  # 설정 정보 로드
from utils.llm_factory import get_llm, get_embeddings, log_llm_error  # LLM/임베딩 생성 및 오류 기록
from utils.vector_store import VectorStoreManager  # 벡터 DB 검색 매니저


# =============================================================================
# 1. State 정의
# =============================================================================

class EntityRAGState(TypedDict):
    """
    Entity RAG 상태

    병렬 실행을 위해 각 검색 결과를 별도의 필드로 관리합니다.
    """
    question: str                    # 사용자 질문
    entities: List[str]              # 추출된 엔티티 이름 리스트
    entity_docs: List[Document]      # 엔티티 검색 결과 (Exact Match)
    semantic_docs: List[Document]    # 의미론적 검색 결과 (Vector Search)
    merged_docs: List[Document]      # 병합된 문서 리스트
    answer: str                      # 최종 답변


# =============================================================================
# 2. Vector Store 준비 (메타데이터 포함)
# =============================================================================

def get_vector_store() -> VectorStoreManager:
    """Vector Store 초기화 및 메타데이터 포함 데이터 로드"""
    embeddings = get_embeddings()
    manager = VectorStoreManager(embeddings=embeddings, collection_name="entity_rag")

    if True: # 항상 데이터 로드 시도 (예제용)
        data = [
            ("LangGraph는 순환 그래프 구조를 지원합니다.", {"tags": "LangGraph"}),
            ("LangChain은 LLM 애플리케이션 프레임워크입니다.", {"tags": "LangChain"}),
            ("RAG는 검색 증강 생성 기술입니다.", {"tags": "RAG"}),
            ("Vector DB는 임베딩을 저장합니다.", {"tags": "VectorDB"}),
        ]
        manager.add_texts([d[0] for d in data], metadatas=[d[1] for d in data])

    return manager


# =============================================================================
# 3. 노드 함수 정의
# =============================================================================

def extract_entities(state: EntityRAGState):
    """엔티티 추출 노드"""
    print(f"\n🏷️ 엔티티 추출 중: {state['question']}")
    
    # 간단한 엔티티 추출 프롬프트 (JSON 출력 유도)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """질문에서 핵심 키워드(엔티티)를 추출하여 JSON 리스트로 반환하세요.
예시: {{"entities": ["Apple", "iPhone"]}}
질문: {question}"""),
    ])
    
    try:
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"question": state["question"]})
        entities = result.get("entities", [])
        print(f"   -> 추출된 엔티티: {entities}")
        return {"entities": entities}
    except Exception as e:
        print(f"   -> 추출 실패, 빈 리스트 반환: {e}")
        return {"entities": []}


def search_by_entity(state: EntityRAGState):
    """엔티티 기반 검색 노드 (병렬 실행 1)"""
    print("🔍 엔티티 검색 수행...")
    vs = get_vector_store()
    results = []
    
    # 추출된 엔티티가 메타데이터나 본문에 포함된 문서 검색 (여기서는 단순 키워드 검색 시뮬레이션)
    for entity in state["entities"]:
        # 실제로는 메타데이터 필터링 등을 사용할 수 있음
        docs = vs.search(entity, k=1)
        results.extend(docs)

    return {"entity_docs": results}


def search_semantic(state: EntityRAGState):
    """의미론적 검색 노드 (병렬 실행 2)"""
    print("🔍 의미론적 검색 수행...")
    vs = get_vector_store()
    docs = vs.search(state["question"], k=2)
    return {"semantic_docs": docs}


def merge_results(state: EntityRAGState):
    """검색 결과 병합 노드"""
    print("🔄 검색 결과 병합 중...")
    
    # 중복 제거 및 병합
    seen = set()
    merged = []
    
    # 엔티티 검색 결과 우선
    for doc in state.get("entity_docs", []) + state.get("semantic_docs", []):
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)

    print(f"   -> 총 {len(merged)}개 문서 병합됨")
    return {"merged_docs": merged}


def generate_answer(state: EntityRAGState):
    """답변 생성 노드"""
    print("📝 답변 생성 중...")
    context = "\n".join(d.page_content for d in state["merged_docs"])
    
    llm = get_llm()
    response = llm.invoke(f"컨텍스트: {context}\n\n질문: {state['question']}\n답변:")
    
    return {"answer": response.content}


# =============================================================================
# 4. 그래프 구성 (병렬 실행)
# =============================================================================

def create_entity_rag_graph():
    """Entity RAG 그래프 생성"""
    builder = StateGraph(EntityRAGState)
    
    # 노드 추가
    builder.add_node("extract_entities", extract_entities)
    builder.add_node("entity_search", search_by_entity)
    builder.add_node("semantic_search", search_semantic)
    builder.add_node("merge", merge_results)
    builder.add_node("generate", generate_answer)
    
    # 엣지 연결
    builder.add_edge(START, "extract_entities")
    
    # 병렬 실행 (Fan-out): extract_entities 완료 후 두 검색 노드로 동시에 분기
    builder.add_edge("extract_entities", "entity_search")
    builder.add_edge("extract_entities", "semantic_search")
    
    # 병합 (Fan-in): 두 검색 노드가 모두 완료되면 merge 노드로 이동
    builder.add_edge("entity_search", "merge")
    builder.add_edge("semantic_search", "merge")
    
    builder.add_edge("merge", "generate")
    builder.add_edge("generate", END)
    
    return builder.compile()


# =============================================================================
# 5. 실행 및 테스트
# =============================================================================

if __name__ == "__main__":
    print("\nLangGraph Entity RAG Example (Parallel Execution)")
    
    graph = create_entity_rag_graph()
    
    questions = ["LangGraph와 LangChain에 대해 알려줘"]
    
    for q in questions:
        print(f"\n{'='*40}\n질문: {q}\n{'='*40}")
        try:
            result = graph.invoke({"question": q})
            print(f"\n🤖 답변: {result['answer']}")
        except Exception as e:
            log_llm_error(e)
