# -*- coding: utf-8 -*-
# 이 파일은 UTF-8 인코딩을 사용하여 한글이 깨지지 않도록 설정합니다. (초심자용 상세 주석 버전)

"""
============================================================================
📚 02a. Rerank RAG - 검색 결과 재정렬(Rerank)하기
============================================================================

단순히 문서들을 찾아오는 것을 넘어, 찾아온 문서들을 AI가 다시 한 번 꼼꼼히 읽고
가장 관련 있는 순서대로 '줄 세우기'를 다시 하는 고급 RAG 기법입니다.

🎯 핵심 학습 포인트:
    1. 2단계 검색: 일단 많이 찾고(Over-fetch), 그중에서 진짜를 고르기(Rerank).
    2. AI 점수 매기기: AI가 각 문서에 0~10점의 점수를 매겨 중요도를 판단합니다.
    3. 정확도 향상: 엉뚱한 문서가 답변에 섞이는 것을 방지합니다.
"""

# =============================================================================
# 📦 필수 라이브러리 임포트 (도구 상자 챙기기)
# =============================================================================

import sys                              # 시스템 관련 도구
import os                               # 환경변수 접근 도구
from pathlib import Path                # 경로 계산 도구
from typing import TypedDict, List      # 데이터 형식 정의용

# 프로젝트의 뿌리(Root) 폴더를 경로에 등록해서 다른 폴더의 파일들을 불러옵니다.
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 파일에서 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# LangChain의 문서 형식과 지시서(프롬프트) 도구
from langchain_openai import ChatOpenAI # LLM 모델 클래스
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# LangGraph의 순서도(그래프) 핵심 도구
from langgraph.graph import StateGraph, START, END

# 프로젝트 전용 AI 모델 호출 도구
from utils.llm_factory import get_embeddings, log_llm_error
from utils.vector_store import VectorStoreManager


# =============================================================================
# 📋 1. 상태(State) 정의하기 (공유 메모장)
# =============================================================================

class RerankRAGState(TypedDict):
    """Rerank RAG가 진행되면서 기록할 정보 목록입니다."""
    question: str                        # 사용자가 던진 질문
    initial_documents: List[Document]    # 1단계에서 대충 많이 찾아온 문서들
    reranked_documents: List[Document]   # 2단계에서 AI가 다시 고른 정예 문서들
    rerank_scores: List[dict]            # AI가 매긴 각 문서의 점수판
    context: str                         # AI에게 전달할 최종 참고 지문
    answer: str                          # AI가 최종적으로 쓴 답변


# =============================================================================
# 🗄️ 2. 지식 저장소(Vector Store) 및 데이터 로더(DataLoader)
# =============================================================================

from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader

def dataloader(manager: VectorStoreManager):
    """./rag 폴더에서 파일을 읽어와 지식 저장소에 적재합니다."""
    print("\n📥 [데이터 로더] ./rag 폴더의 파일들을 지식으로 적재 중...")
    
    # 텍스트 및 CSV 파일 로딩 설정
    documents = []
    # 파일 확장자별 로더 설정 (Windows 안정성을 위해 use_multithreading=False 권장)
    for ext, loader_cls in {".txt": TextLoader, ".md": TextLoader, ".csv": CSVLoader}.items():
        try:
            loader = DirectoryLoader(
                path="./rag", 
                glob=f"**/*{ext}", 
                loader_cls=loader_cls, 
                loader_kwargs={"encoding": "utf-8"}, 
                use_multithreading=False,
                silent_errors=True
            )
            documents.extend(loader.load())
        except: pass

    if documents:
        manager.add_documents(documents)
        print(f"✅ {len(documents)}개의 파일 데이터가 적재되었습니다.")
    else:
        # 파일이 없는 경우 기본 데이터 활용
        samples = [
            "LangGraph는 AI 에이전트의 흐름을 설계하는 도구입니다.",
            "Reranking은 찾은 문서들의 순서를 AI가 다시 정하는 정확도 향상 기술입니다.",
        ]
        manager.add_texts(texts=samples)
        print(f"✅ 기본 데이터 {len(samples)}개가 적재되었습니다. (./rag 폴더가 비어있음)")

def get_rerank_vs() -> VectorStoreManager:
    """Rerank 전용 지식 창고를 만들고 DataLoader를 실행합니다."""
    # 글자를 숫자로 바꿔주는 임베딩 엔진을 가져옵니다.
    embeddings = get_embeddings()
    # 'rerank_rag'라는 이름의 전용 창고를 만듭니다.
    manager = VectorStoreManager(embeddings=embeddings, collection_name="rerank_rag")

    # 데이터 로더를 통해 데이터를 채웁니다.
    dataloader(manager)
    
    return manager


# =============================================================================
# 🔧 3. 각 단계(Node)의 하는 일 정의하기
# =============================================================================

def retrieve_node(state: RerankRAGState) -> dict:
    """[1단계: 일단 많이 찾기] 필요 이상으로 넉넉하게 문서를 검색합니다."""
    print(f"\n🔍 [1단계: 검색] '{state['question']}'와 관련된 문서를 넉넉히(6개) 찾는 중...")
    
    # 지식 창고를 불러옵니다.
    vs = get_rerank_vs()
    # 질문과 닮은 문서를 6개나 찾아옵니다. (나중에 3개로 걸러낼 예정)
    docs = vs.search(query=state["question"], k=6)
    
    print(f"   → 유사한 문서 {len(docs)}개를 일단 확보했습니다.")
    # 찾아온 것들을 'initial_documents' 칸에 보관합니다.
    return {"initial_documents": docs}


def rerank_node(state: RerankRAGState) -> dict:
    """[2단계: AI가 꼼꼼히 다시 고르기] 찾아온 것들 중 진짜 정답 후보를 골라냅니다."""
    print("\n📊 [2단계: 재정렬] AI가 이 문서들을 하나씩 읽고 점수 매기는 중...")
    
    # AI 모델 초기화
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    
    # AI에게 줄 점수 매기기 지침서입니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 아주 엄격한 심사위원입니다.
문서가 사용자의 질문에 얼마나 정확한 대답을 포함하는지 0점에서 10점 사이로 평가하세요.
- 10점: 완벽한 정답!
- 5점: 대충 비슷한 주제임.
- 0점: 전혀 상관없는 소리임.
숫자만 대답하세요."""),
        ("human", "질문: {question}\n문서 내용: {document}\n몇 점입니까? :"),
    ])
    
    scored_docs = [] # 점수가 기록될 임시 목록
    
    # 아까 찾은 6개의 문서를 하나씩 꺼내어 AI에게 물어봅니다.
    for i, doc in enumerate(state["initial_documents"]):
        # AI에게 질문과 문서를 보여주고 점수를 받습니다.
        response = (prompt | model).invoke({
            "question": state["question"],
            "document": doc.page_content
        })
        
        # AI가 말한 텍스트에서 숫자만 뽑아냅니다.
        try:
            score = int(response.content.strip())
        except:
            score = 0 # 에러 나면 0점 처리합니다.
            
        # 문서와 점수를 짝꿍으로 저장합니다.
        scored_docs.append({"document": doc, "score": score})
        print(f"   → [{i+1}번 문서] 심사 점수: {score}점")
    
    # 1. 점수가 높은 순으로 정렬합니다. (내림차순)
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    # 2. 그중에서 1, 2, 3위만 딱 골라냅니다.
    top_3 = scored_docs[:3]
    reranked = [item["document"] for item in top_3]
    
    print(f"   ✅ 선발 완료! 6개 중 가장 우수한 3개 문서만 남겼습니다.")
    
    # 3. 선발된 문서들의 내용을 하나로 합쳐서 나중에 답변할 때 씁니다.
    context_text = "\n\n".join([d.page_content for d in reranked])
    
    return {
        "reranked_documents": reranked,
        "rerank_scores": top_3,
        "context": context_text
    }


def generate_node(state: RerankRAGState) -> dict:
    """[3단계: 답변 쓰기] 엄선된 자료를 바탕으로 질문에 답합니다."""
    print("📝 [3단계: 답변] 최고의 자료들만 모아서 답변을 작성하고 있습니다...")
    
    model = ChatOpenAI(
        base_url=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 제공된 문서를 바탕으로 진실만을 말하는 비서입니다."),
        ("human", "활용할 지식:\n{context}\n\n질문: {question}"),
    ])
    
    # 엄선된 컨텍스트(context)를 사용해 최종 답변을 만듭니다.
    response = (prompt | model).invoke({
        "context": state["context"],
        "question": state["question"]
    })
    
    # 답변 결과를 기록합니다.
    return {"answer": response.content}


# =============================================================================
# 🔗 4. 전체 흐름도(Graph) 짜기
# =============================================================================

def create_graph():
    """Rerank RAG가 어떤 순서로 동작할지 지도를 그립니다."""
    # 우리가 만든 메모장(RerankRAGState)을 사용하는 순서도 캔버스입니다.
    builder = StateGraph(RerankRAGState)
    
    # 1. 각 단계별 기능을 노드로 추가합니다.
    builder.add_node("retrieve", retrieve_node) # 검색 단계
    builder.add_node("rerank", rerank_node)     # 재정렬 단계
    builder.add_node("generate", generate_node) # 답변 생성 단계
    
    # 2. 화살표를 이어 순서를 정합니다.
    builder.add_edge(START, "retrieve")   # 시작 -> 검색
    builder.add_edge("retrieve", "rerank") # 검색 -> 재정렬
    builder.add_edge("rerank", "generate") # 재정렬 -> 답변
    builder.add_edge("generate", END)      # 답변 -> 끝!
    
    # 3. 조립이 끝난 흐름도를 실행 가능하게 만듭니다.
    return builder.compile()


# =============================================================================
# ▶️ 5. 실제로 실행해보기 (CLI)
# =============================================================================

def run_interactive_rerank(question: str, app):
    """사용자가 질문을 치면 이 함수가 작동하여 과정을 보여줍니다."""
    print(f"\n{'='*60}")
    print(f"🙋 질문: {question}")
    print(f"{'='*60}")
    
    try:
        # 흐름도를 가동(invoke)합니다.
        result = app.invoke({
            "question": question,
            "initial_documents": [],
            "reranked_documents": [],
            "rerank_scores": [],
            "context": "",
            "answer": ""
        })
        
        # AI의 최종 답변을 화면에 띄웁니다.
        print(f"\n🤖 최종 답변: {result['answer']}")
        
    except Exception as e:
        log_llm_error(e)
        print(f"❌ 도중에 에러가 났습니다: {e}")


if __name__ == "__main__":
    print("\n" + "📚 Rerank RAG 예제를 시작합니다! (초심자용)")
    print("단순 검색보다 훨씬 똑똑하게 필요한 정보를 골라냅니다.")
    print("- 끝내려면 'q' 또는 'exit'를 입력하세요.\n")
    
    # 1. 지도가 그려진 흐름도를 완성합니다.
    app = create_graph()
    
    # 2. 계속해서 질문을 받습니다.
    while True:
        try:
            query = input("🙋 궁금한 것을 물어보세요: ").strip()
            
            if not query: continue
                
            if query.lower() in ("quit", "exit", "q"):
                print("👋 이용해 주셔서 감사합니다. 다음에 또 봐요!")
                break
                
            # 질문으로 Rerank RAG 가동!
            run_interactive_rerank(query, app)
            
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n⚠️ 예상치 못한 오류 발생: {e}")
            break
