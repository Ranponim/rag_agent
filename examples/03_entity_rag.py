# -*- coding: utf-8 -*-
"""
03. Entity RAG 예제 - 엔티티 기반 RAG 구현

쿼리에서 엔티티를 추출하고, 엔티티 기반 검색과 의미론적 검색을 결합합니다.

학습 목표:
    1. LLM을 활용한 엔티티 추출
    2. 하이브리드 검색 전략 (엔티티 + 의미론적)
    3. 병렬 노드 실행

실행: python examples/03_entity_rag.py
"""

import sys
from pathlib import Path
from typing import TypedDict, List

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

class EntityRAGState(TypedDict):
    """Entity RAG 상태"""
    question: str                    # 사용자 질문
    entities: List[dict]             # 추출된 엔티티 [{"name": str, "type": str}]
    entity_documents: List[Document] # 엔티티 기반 검색 결과
    semantic_documents: List[Document]  # 의미론적 검색 결과
    merged_documents: List[Document] # 병합된 문서
    context: str                     # 최종 컨텍스트
    answer: str                      # 생성된 답변


# =============================================================================
# 2. Vector Store 초기화
# =============================================================================

_entity_vs: VectorStoreManager = None

def get_entity_vs() -> VectorStoreManager:
    """엔티티 Vector Store 반환 (싱글톤)"""
    global _entity_vs
    if _entity_vs is None:
        print("📚 Entity Vector Store 초기화...")
        _entity_vs = VectorStoreManager(
            embeddings=get_embeddings(),
            collection_name="entity_rag",
            chunk_size=300,
        )
        # 샘플 데이터
        samples = [
            ("LangGraph는 LangChain 팀이 개발한 상태 기반 에이전트 프레임워크입니다.", "LangGraph,LangChain"),
            ("RAG는 검색 증강 생성으로, 외부 지식으로 LLM 응답을 개선합니다.", "RAG,LLM"),
            ("ChromaDB는 LangChain과 함께 사용되는 오픈소스 벡터 DB입니다.", "ChromaDB,LangChain"),
            ("OpenAI는 GPT-4와 ChatGPT를 개발한 AI 연구 회사입니다.", "OpenAI,GPT-4,ChatGPT"),
            ("Self-RAG는 LLM이 검색 필요성을 스스로 판단하는 기법입니다.", "Self-RAG,LLM"),
            ("Corrective RAG는 검색 결과 품질을 평가하고 재검색하는 패턴입니다.", "Corrective RAG"),
        ]
        _entity_vs.add_texts(
            texts=[s[0] for s in samples],
            metadatas=[{"entities": s[1]} for s in samples]
        )
        print(f"✅ {len(samples)}개 문서 추가 완료")
    return _entity_vs


# =============================================================================
# 3. 노드 함수
# =============================================================================

def extract_entities_node(state: EntityRAGState) -> dict:
    """쿼리에서 엔티티 추출 (LLM 사용)"""
    print(f"\n🏷️ 엔티티 추출: '{state['question']}'")
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """텍스트에서 기술/개념/조직 엔티티를 추출하세요.
JSON 형식: {{"entities": [{{"name": "이름", "type": "technology|concept|organization"}}]}}
엔티티 없으면: {{"entities": []}}"""),
        ("human", "{question}"),
    ])
    
    try:
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"question": state["question"]})
        entities = result.get("entities", [])
        print(f"   → 추출: {[e['name'] for e in entities]}")
    except Exception as e:
        print(f"   ⚠️ 추출 실패: {e}")
        entities = []
    
    return {"entities": entities}


def entity_search_node(state: EntityRAGState) -> dict:
    """엔티티 기반 문서 검색"""
    print("\n🔍 엔티티 기반 검색...")
    
    entities = state.get("entities", [])
    if not entities:
        return {"entity_documents": []}
    
    vs = get_entity_vs()
    entity_docs = []
    
    for entity in entities:
        docs = vs.search(query=entity["name"], k=2)
        for doc in docs:
            if entity["name"].lower() in doc.metadata.get("entities", "").lower():
                if doc not in entity_docs:
                    entity_docs.append(doc)
    
    print(f"   → {len(entity_docs)}개 문서")
    return {"entity_documents": entity_docs}


def semantic_search_node(state: EntityRAGState) -> dict:
    """의미론적 검색"""
    print("\n🔎 의미론적 검색...")
    
    docs = get_entity_vs().search(query=state["question"], k=3)
    print(f"   → {len(docs)}개 문서")
    return {"semantic_documents": docs}


def merge_results_node(state: EntityRAGState) -> dict:
    """검색 결과 병합 (엔티티 우선, 중복 제거)"""
    print("\n🔀 결과 병합...")
    
    entity_docs = state.get("entity_documents", [])
    semantic_docs = state.get("semantic_documents", [])
    
    # 엔티티 문서 우선, 중복 제거
    merged = list(entity_docs)
    seen = {doc.page_content for doc in merged}
    
    for doc in semantic_docs:
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    
    merged = merged[:5]  # 최대 5개
    
    # 컨텍스트 생성
    context = "\n\n".join([
        f"[문서 {i+1}] {doc.page_content}" for i, doc in enumerate(merged)
    ])
    
    print(f"   → 최종 {len(merged)}개")
    return {"merged_documents": merged, "context": context}


def generate_answer_node(state: EntityRAGState) -> dict:
    """답변 생성"""
    print("\n💭 답변 생성...")
    
    llm = get_llm()
    entities_str = ", ".join([e["name"] for e in state.get("entities", [])]) or "없음"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """컨텍스트 기반으로 답변하세요.
주요 엔티티: {entities}
컨텍스트:
{context}"""),
        ("human", "{question}"),
    ])
    
    response = (prompt | llm).invoke({
        "entities": entities_str,
        "context": state["context"],
        "question": state["question"],
    })
    
    return {"answer": response.content}


# =============================================================================
# 4. 그래프 생성
# =============================================================================

def create_entity_rag_graph():
    """
    Entity RAG 그래프 생성
    
    구조:
        START → extract_entities → entity_search ─┐
                                                  ├→ merge → generate → END
                                 semantic_search ─┘
    """
    graph = StateGraph(EntityRAGState)
    
    # 노드 추가
    graph.add_node("extract_entities", extract_entities_node)
    graph.add_node("entity_search", entity_search_node)
    graph.add_node("semantic_search", semantic_search_node)
    graph.add_node("merge", merge_results_node)
    graph.add_node("generate", generate_answer_node)
    
    # 엣지: 시작 → 엔티티 추출 → 병렬 검색 → 병합 → 생성 → 종료
    graph.add_edge(START, "extract_entities")
    graph.add_edge("extract_entities", "entity_search")
    graph.add_edge("extract_entities", "semantic_search")
    graph.add_edge("entity_search", "merge")
    graph.add_edge("semantic_search", "merge")
    graph.add_edge("merge", "generate")
    graph.add_edge("generate", END)
    
    print("✅ Entity RAG 그래프 컴파일 완료!")
    return graph.compile()


# =============================================================================
# 5. 실행
# =============================================================================

def run_entity_rag(question: str) -> str:
    """Entity RAG 실행"""
    graph = create_entity_rag_graph()
    
    initial_state = {
        "question": question, "entities": [], "entity_documents": [],
        "semantic_documents": [], "merged_documents": [], "context": "", "answer": "",
    }
    
    print(f"\n{'='*60}\n🙋 질문: {question}\n{'='*60}")
    result = graph.invoke(initial_state)
    
    print(f"\n🏷️ 엔티티: {[e['name'] for e in result['entities']]}")
    print(f"📚 최종 문서: {len(result['merged_documents'])}개")
    print(f"\n🤖 답변:\n{result['answer']}\n{'='*60}")
    
    return result["answer"]


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Entity RAG 예제")
    print("="*60)
    # 설정 확인 (제거됨: Local LLM 등 다양한 환경 지원을 위해 엄격한 키 검증 생략)
    
    test_queries = [
        "LangGraph와 LangChain의 관계는?",
        "Self-RAG와 Corrective RAG의 차이점은?",
    ]
    
    from utils.llm_factory import log_llm_error
    
    for query in test_queries:
        try:
            run_entity_rag(query)
        except Exception as e:
            # 오류 발생 시 상세 로깅
            log_llm_error(e)
            print(f"❌ 오류 발생: {e}")
        print()
