# -*- coding: utf-8 -*-
# 이 파일은 UTF-8 인코딩을 사용하여 한글이 깨지지 않도록 설정합니다. (초심자용 상세 주석 버전)

"""
============================================================================
📚 04a. Adaptive RAG - 질문의 난이도에 맞춰 스스로 변하는 AI
============================================================================

사용자가 물어보는 질문이 '쉬운지', '보통인지', '어려운지' AI가 먼저 판단하고, 
그 난이도에 가장 적합한 검색 전략을 자동으로 선택하는 '적응형 RAG' 기법입니다.

🎯 핵심 학습 포인트:
    1. 질문 분류: 질문을 simple(쉬움), moderate(보통), complex(어려움)로 나눕니다.
    2. 동적 경로: 난이도에 따라 서로 다른 처리 과정(노드)으로 안내합니다.
    3. 효율성: 쉬운 건 바로 답해서 아끼고, 어려운 건 깊게 조사해서 정확도를 높입니다.
"""

# =============================================================================
# 📦 필수 라이브러리 임포트 (도구함 열기)
# =============================================================================

import sys                              # 시스템 환경 제어
import os                               # 환경변수 접근용
from pathlib import Path                # 파일 경로 처리
from typing import TypedDict, List, Literal  # 데이터 형식 및 리터럴 타입 정의

# 프로젝트 최상위 폴더를 인식시켜 config나 utils를 사용할 수 있게 합니다.
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# LangChain 문서 형식 및 프롬프트 도구
from langchain_openai import ChatOpenAI # LLM 모델 클래스
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# LangGraph 순서도(그래프) 제작 도구
from langgraph.graph import StateGraph, START, END

# 프로젝트 전용 유틸리티
from utils.llm_factory import get_embeddings, log_llm_error
from utils.vector_store import VectorStoreManager


# =============================================================================
# 📋 1. 상태(State) 정의하기 (공유 작업노트)
# =============================================================================

class AdaptiveRAGState(TypedDict):
    """일의 진행 상황을 기록할 항목들입니다."""
    question: str                    # 사용자가 던진 질문
    query_complexity: str            # AI가 판별한 질문의 난이도 (쉬움/보통/어려움)
    strategy_used: str               # 이번에 어떤 전략을 썼는지 기록 (확인용)
    documents: List[Document]        # 지식 창고에서 찾은 결과들
    context: str                     # 답변을 위해 정리된 참고 지식
    answer: str                      # AI가 내놓은 최종 답변


# =============================================================================
# 🗄️ 2. Vector Store 초기화 (공통 모듈 사용)
# =============================================================================

from utils.data_loader import get_rag_vector_store

def get_adaptive_vs() -> VectorStoreManager:
    """적응형 RAG를 위한 Vector Store를 준비합니다."""
    return get_rag_vector_store(collection_name="rag_collection")


# =============================================================================
# 🧠 3. 관문 노드: 질문의 난이도 판별 (Classification)
# =============================================================================

def classify_query_node(state: AdaptiveRAGState) -> dict:
    """[판별 단계] 질문을 읽고 '쉬움/보통/어려움' 중 하나로 분류합니다."""
    print(f"\n🧐 [분류] 질문의 수준을 분석 중입니다... 어떤 전략이 좋을까요?")
    
    # AI 모델 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    # 심사위원 AI에게 질문의 난이도를 판단해달라고 지시합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 질문 분석 전문가입니다. 다음 3가지 중 하나로만 대답하세요.
1. "simple": 인사, 이름 묻기, 혹은 아주 뻔한 상식 질문
2. "moderate": 지식 창고 검색이 한 번쯤 필요한 일반적인 질문
3. "complex": 여러 관점의 분석, 비교, 깊은 사고가 필요한 복잡한 질문
오직 영문 단어 하나("simple", "moderate", "complex")만 답변하세요."""),
        ("human", "사용자 질문: {question}"),
    ])
    
    response = (prompt | model).invoke({"question": state["question"]})
    # AI의 답변을 소문자로 바꾸고 공백을 제거합니다.
    complexity = response.content.lower().strip()
    
    # 만약 AI가 이상한 말을 하면 기본값으로 '보통(moderate)'을 지정합니다.
    if complexity not in ["simple", "moderate", "complex"]:
        complexity = "moderate"
        
    print(f"   → 판단 결과: 이 질문은 '{complexity}' 수준입니다.")
    # 판단 결과를 기록합니다.
    return {"query_complexity": complexity}


# =============================================================================
# 🛠️ 4. 전략별 행동 요강 (각 단계 정의)
# =============================================================================

def simple_strategy_node(state: AdaptiveRAGState) -> dict:
    """[전략 1: 쉬운 질문] 검색 없이 AI 본인의 상식으로 바로 답합니다."""
    print("⚡ [Simple] 너무 쉬운 질문이라 검색 없이 바로 대답합니다.")
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    res = model.invoke(state["question"])
    return {"strategy_used": "Simple (직접 답변)", "answer": res.content}


def moderate_strategy_node(state: AdaptiveRAGState) -> dict:
    """[전략 2: 보통 질문] 지식 창고에서 자료를 한 번 찾아보고 답합니다."""
    print("📚 [Moderate] 지식 창고에서 필요한 자료를 한 번 찾아봅니다.")
    vs = get_adaptive_vs()
    # 질문과 닮은 자료를 3개 찾아옵니다.
    docs = vs.search(state["question"], k=3)
    
    # 찾은 자료들을 한데 묶습니다.
    context = "\n".join([d.page_content for d in docs])
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    # 찾은 자료와 함께 질문을 던져 답변을 받습니다.
    res = model.invoke(f"지식 내용:\n{context}\n\n질문: {state['question']}")
    
    return {
        "strategy_used": "Moderate (일반 RAG)", 
        "documents": docs, 
        "answer": res.content
    }


def complex_strategy_node(state: AdaptiveRAGState) -> dict:
    """[전략 3: 어려운 질문] 질문을 쪼개서 깊게 조사하고 분석 보고서를 씁니다."""
    print("🔬 [Complex] 질문이 복잡하네요! 여러 단계로 나눠서 정밀 분석합니다.")
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    # 1. 어려운 질문을 해결하기 위한 2개의 세부 질문을 AI에게 먼저 물어봅니다.
    decompose_res = model.invoke(f"이 어려운 질문을 해결하기 위해 먼저 알아야 할 기초 질문 2개만 뽑아주세요. 한 줄씩 쓰세요.\n질문: {state['question']}")
    sub_queries = [q.strip() for q in decompose_res.content.split("\n") if q.strip()][:2]
    
    print(f"   → 단계별 세부 조사 항목: {sub_queries}")
    
    # 2. 세부 질문들로 각각 지식 창고를 뒤집니다.
    vs = get_adaptive_vs()
    all_context = []
    for sq in sub_queries + [state["question"]]:
        docs = vs.search(sq, k=2)
        all_context.extend([d.page_content for d in docs])
    
    # 3. 모은 모든 정보를 합쳐서(중복 제거) 심층 보고서 형태의 답변을 생성합니다.
    final_context = "\n".join(list(set(all_context)))
    res = model.invoke(f"심층 분석 답변 요청:\n관련된 모든 정보:\n{final_context}\n\n최종 질문: {state['question']}")
    
    return {
        "strategy_used": "Complex (다단계 정밀 RAG)", 
        "answer": res.content
    }


# =============================================================================
# 🚦 5. 신호등(라우터) 및 전체 지도(Graph) 만들기
# =============================================================================

def route_complexity(state: AdaptiveRAGState) -> Literal["simple", "moderate", "complex"]:
    """AI가 판단한 난이도 칸을 보고 어느 길로 갈지 안내합니다."""
    return state["query_complexity"]

def create_graph():
    """상황에 따라 길이 바뀌는 '똑똑한 지도'를 완성합니다."""
    # 우리가 만든 작업노트(AdaptiveRAGState)를 사용하는 순서도입니다.
    builder = StateGraph(AdaptiveRAGState)
    
    # 1. 할 일(노드)들을 등록합니다.
    builder.add_node("classify", classify_query_node) # 판별사
    builder.add_node("simple", simple_strategy_node)   # 쉬운 길
    builder.add_node("moderate", moderate_strategy_node) # 보통 길
    builder.add_node("complex", complex_strategy_node)   # 어려운 길
    
    # 2. 시작 전에는 무조건 '판별사'에게 보내줍니다.
    builder.add_edge(START, "classify")
    
    # 3. 판별사가 정한 난이도에 따라 세 갈래 길로 나눠 보냅니다. (조건부 연결)
    builder.add_conditional_edges(
        "classify",
        route_complexity, # 신호등 역할 함수
        {
            "simple": "simple",
            "moderate": "moderate",
            "complex": "complex"
        }
    )
    
    # 4. 어떤 길로 가든 마지막엔 대화가 끝납니다(END).
    builder.add_edge("simple", END)
    builder.add_edge("moderate", END)
    builder.add_edge("complex", END)
    
    # 5. 완성된 지도를 실행기에 넣습니다.
    return builder.compile()


# =============================================================================
# ▶️ 6. 실제로 돌려보기 (실행 프로그램)
# =============================================================================

def run_adaptive_rag(query: str, app):
    """질문을 하면 AI가 난이도를 분석하고 그에 맞춰 답변해줍니다."""
    print(f"\n{'='*60}")
    print(f"🙋 질문: {query}")
    print(f"{'='*60}")
    
    try:
        # 가동 준비 및 초기 메모장 세팅
        result = app.invoke({
            "question": query,
            "query_complexity": "",
            "strategy_used": "",
            "documents": [],
            "context": "",
            "answer": ""
        })
        
        # 어떤 전략을 골랐는지와 최종 답변을 보여줍니다.
        print(f"\n📊 선택된 전략: {result['strategy_used']}")
        print(f"\n🤖 AI의 답변:\n{result['answer']}")
        
    except Exception as e:
        log_llm_error(e)
        print(f"❌ 도중에 시스템 오류가 났습니다: {e}")


if __name__ == "__main__":
    print("\n" + "🌟 상황 맞춤형 Adaptive RAG를 가동합니다! 🌟")
    print("질문의 난이도를 AI가 스스로 판단하여 가장 효율적으로 일합니다.")
    print("- 종료하려면 'q' 혹은 'exit'를 입력하세요.\n")
    
    # 1. 뼈대가 되는 흐름도 기계를 완성합니다.
    app = create_graph()
    
    # 2. 질문을 계속 받습니다.
    while True:
        try:
            user_input = input("🙋 어떤 것이든 물어보세요 : ").strip()
            
            if not user_input: continue
                
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 똑똑한 대화를 즐겨주셔서 감사합니다! 안녕히 가세요.")
                break
                
            # 질문으로 시스템 작동!
            run_adaptive_rag(user_input, app)
            
        except KeyboardInterrupt:
            print("\n👋 급히 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n⚠️ 시스템 오류 발생: {e}")
            break
