# -*- coding: utf-8 -*-
# 이 파일은 UTF-8 인코딩을 사용하여 한글이 깨지지 않도록 설정합니다. (초심자용 상세 주석 버전)

"""
============================================================================
📚 02b. Query Transform RAG - 질문(Query)을 더 똑똑하게 바꿔서 검색하기
============================================================================

사용자가 대충 물어봐도 AI가 그 질문을 검색하기 좋은 형태로 '변신(Transform)'시켜
더 정확한 정보를 찾아내는 고급 기술들을 배웁니다.

🎯 핵심 학습 포인트:
    1. HyDE: 질문에 대한 '가짜 대답'을 먼저 상상해보고, 그 상상을 바탕으로 검색합니다.
    2. Multi-Query: 질문 하나를 3~4개의 다양한 표현으로 바꿔서 그물망을 넓게 펼칩니다.
    3. 병렬 검색: 여러 갈래의 검색을 동시에 진행하여 시간을 단축하고 정확도를 높입니다.
"""

# =============================================================================
# 📦 필수 라이브러리 임포트 (도구 가방 챙기기)
# =============================================================================

import sys                              # 시스템 환경 제어
import os                               # 환경변수 접근용
from pathlib import Path                # 파일 경로 처리
from typing import TypedDict, List      # 데이터 형식 정의

# 프로젝트 최상단 폴더를 경로에 추가하여 config, utils 등을 불러옵니다.
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# LangChain의 문서 형식과 지시서(프롬프트) 도구
from langchain_openai import ChatOpenAI # LLM 모델 클래스
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# LangGraph의 순서도(그래프) 제작 도구
from langgraph.graph import StateGraph, START, END

# 프로젝트 전용 유틸리티들
from utils.llm_factory import get_embeddings, log_llm_error
from utils.vector_store import VectorStoreManager


# =============================================================================
# 📋 1. 상태(State) 정의하기 (공유 작업판)
# =============================================================================

class QueryTransformState(TypedDict):
    """이 RAG 시스템이 일하면서 적어둘 메모장 항목들입니다."""
    original_question: str               # 사용자가 입력한 원래 질문
    hyde_document: str                   # 1. HyDE 기법으로 만든 '가상 답변' 지문
    multi_queries: List[str]             # 2. 여러 관점으로 다시 쓴 질문 목록
    hyde_results: List[Document]         # HyDE로 찾아낸 실제 문서들
    multi_query_results: List[Document]  # 변형 질문들로 찾아낸 실제 문서들
    merged_documents: List[Document]     # 모든 검색 결과를 하나로 합친 목록
    context: str                         # AI에게 보여줄 최종 참고 지문 합본
    answer: str                          # AI가 최종적으로 작성한 답변


# =============================================================================
# 🗄️ 2. Vector Store 초기화 (공통 모듈 사용)
# =============================================================================

from utils.data_loader import get_rag_vector_store

def get_qt_vs() -> VectorStoreManager:
    """검색 변환 전용 지식 창고를 생성하고 데이터를 로드합니다."""
    return get_rag_vector_store(collection_name="query_transform_rag")


# =============================================================================
# 🔧 3. 각 단계(Node)에서 하는 일 정의하기
# =============================================================================

def generate_hyde_document(state: QueryTransformState) -> dict:
    """[경로 A-1] HyDE 가상 문서 만들기: '답변은 이럴 거야'라고 상상하기"""
    print(f"\n🔮 [HyDE] 질문에 대한 '가상의 정답'을 상상해서 써보는 중...")
    
    # AI 모델 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    # AI에게 가짜 답변을 아주 유식하게 써달라고 부탁합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 지식 백과사전 편집자입니다. 질문에 대해 아주 상세하고 전문적인 '가상 답변'을 한 문단으로 작성하세요."),
        ("human", "{question}"),
    ])
    
    # AI가 상상한 답변을 생성합니다.
    response = (prompt | model).invoke({"question": state["original_question"]})
    print(f"   → 가상 답변 상상 완료! 이를 바탕으로 검색을 시작합니다.")
    
    # 생성된 가상 답변을 'hyde_document' 칸에 적습니다.
    return {"hyde_document": response.content}


def generate_multi_queries(state: QueryTransformState) -> dict:
    """[경로 B-1] Multi-Query 만들기: 질문을 여러 방식으로 다시 쓰기"""
    print(f"\n🔄 [Multi-Query] 질문을 3가지 다른 표현으로 변형하는 중...")
    
    # AI 모델 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    # 질문의 의미는 같지만 단어 구성을 다르게 하여 검색 그물을 넓힙니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "원본 질문을 바탕으로 검색에 도움이 될만한 변형 질문 3개를 만드세요. 한 줄에 하나씩만 쓰세요."),
        ("human", "원본 질문: {question}"),
    ])
    
    response = (prompt | model).invoke({"question": state["original_question"]})
    
    # AI의 답변을 줄 단위로 쪼개 리스트로 만듭니다.
    queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    # 원본 질문까지 포함해서 총 4개의 질문 리스트를 확보합니다.
    final_queries = [state["original_question"]] + queries[:3]
    
    print(f"   → 확장된 질문 그물: {final_queries}")
    # 여러 질문들을 'multi_queries' 칸에 기록합니다.
    return {"multi_queries": final_queries}


def search_with_hyde(state: QueryTransformState) -> dict:
    """[경로 A-2] 상상한 답변(HyDE)과 가장 비슷한 진짜 문서 찾기"""
    print(f"🔍 [HyDE 검색] AI의 상상력과 가장 일치하는 진짜 자료를 찾는 중...")
    vs = get_qt_vs()
    # 가짜 답변을 쿼리로 써서 실제 지식 창고를 뒤집니다.
    docs = vs.search(query=state["hyde_document"], k=3)
    return {"hyde_results": docs}


def search_with_multi_queries(state: QueryTransformState) -> dict:
    """[경로 B-2] 4개의 질문 그물로 싹쓸이 검색하기"""
    print(f"🔍 [Multi-Query 검색] {len(state['multi_queries'])}개의 질문 그물로 넓게 뒤지는 중...")
    vs = get_qt_vs()
    
    all_docs = []
    seen_content = set() # 중복된 내용을 걸러내기 위한 장치
    
    # 각 질문마다 돌아가며 검색합니다.
    for q in state["multi_queries"]:
        docs = vs.search(query=q, k=2)
        for d in docs:
            # 이미 찾은 내용이 아니면 목록에 담습니다.
            if d.page_content not in seen_content:
                all_docs.append(d)
                seen_content.add(d.page_content)
                
    return {"multi_query_results": all_docs}


def merge_results(state: QueryTransformState) -> dict:
    """[통합 단계] 두 경로(A, B)에서 얻은 문서들을 하나로 예쁘게 합치기"""
    print(f"\n🔀 [결과 합치기] 모든 검색 경로의 결과를 통합하고 중복을 제거합니다.")
    
    seen = set()
    merged = []
    
    # HyDE 검색 결과와 Multi-Query 검색 결과를 한 통에 담습니다.
    total_docs = state.get("hyde_results", []) + state.get("multi_query_results", [])
    
    for doc in total_docs:
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    
    # 너무 복잡하면 상위 5개만 최종 후보로 정합니다.
    final_docs = merged[:5]
    print(f"   → 최종적으로 {len(final_docs)}개의 유니크한 지식 문서를 확보했습니다.")
    
    # AI가 읽기 좋게 문장들을 합쳐서 컨텍스트로 만듭니다.
    context = "\n\n".join([f"[참조{i+1}] {d.page_content}" for i, d in enumerate(final_docs)])
    
    return {"merged_documents": final_docs, "context": context}


def generate_answer(state: QueryTransformState) -> dict:
    """[마지막: 답변 쓰기] 풍부하게 모은 지식으로 완벽한 답장 쓰기"""
    print("📝 [최종 답변] 정교하게 수집된 정보들을 바탕으로 답변을 작성합니다...")
    
    # AI 모델 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 도서관 사서처럼 정확한 정보만을 알려주는 AI 가이드입니다."),
        ("human", "참조한 지식들:\n{context}\n\n사용자 질문: {question}"),
    ])
    
    # 모든 정보를 종합하여 답변을 생성합니다.
    response = (prompt | model).invoke({
        "context": state["context"],
        "question": state["original_question"]
    })
    
    # 드디어 완성된 답변을 기록합니다.
    return {"answer": response.content}


# =============================================================================
# 🔗 4. 전체적인 업무 흐름도(Graph) 조립하기
# =============================================================================

def create_graph():
    """병렬(동시) 검색이 가능한 고급 RAG 순서도를 만듭니다."""
    # 우리가 만든 메모장(QueryTransformState)을 사용하는 도면을 펼칩니다.
    builder = StateGraph(QueryTransformState)
    
    # 1. 일할 사람(노드)들을 이름표와 함께 등록합니다.
    builder.add_node("gen_hyde", generate_hyde_document)
    builder.add_node("gen_multi", generate_multi_queries)
    builder.add_node("search_hyde", search_with_hyde)
    builder.add_node("search_multi", search_with_multi_queries)
    builder.add_node("merge", merge_results)
    builder.add_node("generate", generate_answer)
    
    # 2. 화살표를 이어줍니다. (START에서 두 갈래로 나뉩니다!)
    builder.add_edge(START, "gen_hyde")             # A경로: HyDE 시작
    builder.add_edge(START, "gen_multi")            # B경로: Multi-Query 시작
    
    builder.add_edge("gen_hyde", "search_hyde")     # A경로 이어가기
    builder.add_edge("gen_multi", "search_multi")   # B경로 이어가기
    
    builder.add_edge("search_hyde", "merge")        # A결과를 합치기 단계로 보냄
    builder.add_edge("search_multi", "merge")       # B결과도 합치기 단계로 보냄
    
    builder.add_edge("merge", "generate")           # 합쳐진 결과로 답변 시작
    builder.add_edge("generate", END)               # 답변 끝!
    
    # 3. 조립 완료된 순서도를 실행 가능한 기계(Graph)로 만듭니다.
    return builder.compile()


# =============================================================================
# ▶️ 5. 실제로 돌려보기 (CLI 실행부)
# =============================================================================

def run_qt_rag(query: str, app):
    """질문을 입력하면 작동 과정을 보여주며 답변합니다."""
    print(f"\n{'='*60}")
    print(f"🙋 질문: {query}")
    print(f"{'='*60}")
    
    try:
        # 가동 준비(입력값 세팅)
        result = app.invoke({
            "original_question": query,
            "hyde_document": "",
            "multi_queries": [],
            "hyde_results": [],
            "multi_query_results": [],
            "merged_documents": [],
            "context": "",
            "answer": ""
        })
        
        # 탄생한 답변을 보여줍니다.
        print(f"\n🤖 AI 가이드의 답변:\n{result['answer']}")
        
    except Exception as e:
        log_llm_error(e)
        print(f"❌ 도중에 시스템 오류가 났습니다: {e}")


if __name__ == "__main__":
    print("\n" + "🌟 Query Transform RAG 시스템을 가동합니다! 🌟")
    print("질문을 어떻게 바꿔서 검색하는지 과정을 지켜보세요.")
    print("- 종료하려면 'q' 혹은 'exit'를 입력하세요.\n")
    
    # 1. 흐름도 기계를 한 번 만들어 둡니다.
    app = create_graph()
    
    # 2. 반복해서 질문을 받습니다.
    while True:
        try:
            line = input("🙋 검색하고 싶은 것을 적어주세요: ").strip()
            
            if not line: continue
                
            if line.lower() in ("quit", "exit", "q"):
                print("👋 이용해 주셔서 감사합니다! 좋은 하루 되세요.")
                break
                
            # 실행!
            run_qt_rag(line, app)
            
        except KeyboardInterrupt:
            print("\n👋 급히 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n⚠️ 예기치 못한 에러: {e}")
            break
