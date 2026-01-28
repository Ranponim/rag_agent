# -*- coding: utf-8 -*-
"""
Vector Store 관리 모듈

이 모듈은 RAG 시스템에서 사용하는 Vector Store를 관리합니다.
ChromaDB를 기본 백엔드로 사용하며, 문서 임베딩 및 유사도 검색 기능을 제공합니다.

주요 기능:
    - 문서 로드 및 청킹
    - Vector Store 생성 및 관리
    - 유사도 기반 문서 검색

사용 예시:
    from utils.vector_store import VectorStoreManager
    
    # Vector Store 매니저 생성
    manager = VectorStoreManager()
    
    # 문서 추가
    manager.add_documents(documents)
    
    # 검색
    results = manager.search("쿼리 텍스트")
"""

import logging
from pathlib import Path
from typing import List, Optional, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Vector Store를 관리하는 클래스
    
    이 클래스는 문서를 청킹하고, 임베딩하여 Vector Store에 저장하는
    전체 파이프라인을 관리합니다.
    
    Attributes:
        embeddings: 임베딩 모델 인스턴스
        vector_store: Vector Store 인스턴스
        text_splitter: 텍스트 분할기
        collection_name: 컬렉션 이름
        persist_directory: 영구 저장 경로
    
    Example:
        >>> from utils.llm_factory import get_embeddings
        >>> embeddings = get_embeddings()
        >>> manager = VectorStoreManager(embeddings=embeddings)
        >>> manager.add_texts(["문서 1", "문서 2"])
        >>> results = manager.search("검색어")
    """
    
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        collection_name: str = "langgraph_rag",
        persist_directory: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        VectorStoreManager 초기화
        
        Args:
            embeddings: 임베딩 모델 (None이면 자동 생성)
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: 영구 저장 경로 (None이면 메모리에만 저장)
            chunk_size: 텍스트 청크 크기
            chunk_overlap: 청크 간 중복 크기
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 임베딩 모델 설정
        if embeddings is None:
            from utils.llm_factory import get_embeddings
            self.embeddings = get_embeddings()
        else:
            self.embeddings = embeddings
        
        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        
        # Vector Store 초기화 (지연 초기화)
        self._vector_store: Optional[VectorStore] = None
        
        logger.info(
            f"VectorStoreManager 초기화 완료 "
            f"(컬렉션: {collection_name}, 청크 크기: {chunk_size})"
        )
    
    @property
    def vector_store(self) -> VectorStore:
        """
        Vector Store 인스턴스를 반환합니다 (지연 초기화).
        
        Returns:
            VectorStore: ChromaDB Vector Store 인스턴스
        """
        if self._vector_store is None:
            self._vector_store = self._create_vector_store()
        return self._vector_store
    
    def _create_vector_store(self) -> VectorStore:
        """
        ChromaDB Vector Store를 생성합니다.
        
        Returns:
            VectorStore: 새로운 ChromaDB 인스턴스
        """
        from langchain_chroma import Chroma
        
        logger.info("ChromaDB Vector Store 생성 중...")
        
        # 영구 저장 설정
        persist_kwargs = {}
        if self.persist_directory:
            persist_kwargs["persist_directory"] = self.persist_directory
            # 디렉토리 생성
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            **persist_kwargs
        )
        
        logger.info("ChromaDB Vector Store 생성 완료")
        return vector_store
    
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할합니다.
        
        Args:
            text: 분할할 텍스트
        
        Returns:
            List[str]: 분할된 텍스트 청크 리스트
        
        Example:
            >>> chunks = manager.split_text("긴 텍스트...")
            >>> print(f"생성된 청크 수: {len(chunks)}")
        """
        chunks = self.text_splitter.split_text(text)
        logger.info(f"텍스트를 {len(chunks)}개의 청크로 분할했습니다.")
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Document 객체들을 청크로 분할합니다.
        
        Args:
            documents: 분할할 Document 리스트
        
        Returns:
            List[Document]: 분할된 Document 리스트
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info(
            f"{len(documents)}개의 문서를 {len(chunks)}개의 청크로 분할했습니다."
        )
        return chunks
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        """
        텍스트를 Vector Store에 추가합니다.
        
        Args:
            texts: 추가할 텍스트 리스트
            metadatas: 각 텍스트의 메타데이터 (선택)
        
        Returns:
            List[str]: 추가된 문서의 ID 리스트
        
        Example:
            >>> ids = manager.add_texts(
            ...     texts=["문서 1", "문서 2"],
            ...     metadatas=[{"source": "doc1"}, {"source": "doc2"}]
            ... )
        """
        # 임베딩 요청 직전 상세 로그 (디버깅용)
        print(f"\n📤 임베딩 모델로 요청 준비 중...")
        print(f"   - 요청할 텍스트 수: {len(texts)}개")
        print(f"   - 임베딩 모델 타입: {type(self.embeddings).__name__}")
        print(f"   - 임베딩 모델 정보: {self.embeddings}")
        
        # 첫 번째 텍스트의 미리보기 (디버깅용)
        if texts:
            preview = texts[0][:100].replace('\n', ' ')
            print(f"   - 첫 번째 텍스트 미리보기: {preview}...")
        
        logger.info(f"{len(texts)}개의 텍스트를 Vector Store에 추가 중...")
        
        try:
            print("   ⏳ 임베딩 모델로 벡터화 요청 중... (서버 응답 대기)")
            ids = self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            print(f"   ✅ 임베딩 완료! {len(ids)}개의 벡터가 생성되었습니다.")
        except Exception as e:
            print(f"\n❌ 임베딩 중 오류 발생!")
            print(f"   오류 타입: {type(e).__name__}")
            print(f"   오류 메시지: {str(e)}")
            raise  # 오류를 다시 던져서 상위에서 처리하도록 함
        
        logger.info(f"{len(ids)}개의 텍스트가 추가되었습니다.")
        return ids
    
    def add_documents(
        self,
        documents: List[Document],
        split: bool = True,
    ) -> List[str]:
        """
        Document 객체들을 Vector Store에 추가합니다.
        
        Args:
            documents: 추가할 Document 리스트
            split: 문서를 청크로 분할할지 여부 (기본값: True)
        
        Returns:
            List[str]: 추가된 문서의 ID 리스트
        """
        if split:
            documents = self.split_documents(documents)
        
        # 임베딩 요청 직전 상세 로그 (디버깅용)
        print(f"\n📤 임베딩 모델로 문서 벡터화 요청 준비 중...")
        print(f"   - 요청할 문서 수: {len(documents)}개")
        print(f"   - 임베딩 모델 타입: {type(self.embeddings).__name__}")
        
        logger.info(f"{len(documents)}개의 문서를 Vector Store에 추가 중...")
        
        try:
            print("   ⏳ 임베딩 모델로 벡터화 요청 중... (서버 응답 대기)")
            ids = self.vector_store.add_documents(documents=documents)
            print(f"   ✅ 임베딩 완료! {len(ids)}개의 벡터가 생성되었습니다.")
        except Exception as e:
            print(f"\n❌ 문서 임베딩 중 오류 발생!")
            print(f"   오류 타입: {type(e).__name__}")
            print(f"   오류 메시지: {str(e)}")
            raise  # 오류를 다시 던져서 상위에서 처리하도록 함
        
        logger.info(f"{len(ids)}개의 문서가 추가되었습니다.")
        return ids
    
    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        유사도 기반으로 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수 (기본값: 4)
            filter: 메타데이터 필터 (선택)
        
        Returns:
            List[Document]: 검색된 Document 리스트
        
        Example:
            >>> results = manager.search("LangGraph란?", k=3)
            >>> for doc in results:
            ...     print(doc.page_content[:100])
        """
        logger.info(f"검색 중: '{query}' (k={k})")
        
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )
        
        logger.info(f"{len(results)}개의 문서를 찾았습니다.")
        return results
    
    def search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> List[tuple[Document, float]]:
        """
        유사도 점수와 함께 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
        
        Returns:
            List[tuple[Document, float]]: (Document, 유사도 점수) 튜플 리스트
        """
        logger.info(f"점수 포함 검색 중: '{query}' (k={k})")
        
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )
        
        logger.info(f"{len(results)}개의 문서를 찾았습니다.")
        return results
    
    def as_retriever(self, **kwargs) -> Any:
        """
        Vector Store를 Retriever로 변환합니다.
        
        LangChain의 LCEL 체인에서 사용할 수 있는 Retriever를 반환합니다.
        
        Args:
            **kwargs: Retriever 설정 (search_type, search_kwargs 등)
        
        Returns:
            VectorStoreRetriever: Retriever 인스턴스
        
        Example:
            >>> retriever = manager.as_retriever(
            ...     search_type="similarity",
            ...     search_kwargs={"k": 4}
            ... )
        """
        return self.vector_store.as_retriever(**kwargs)
    
    def load_from_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> List[str]:
        """
        텍스트 파일에서 문서를 로드하고 Vector Store에 추가합니다.
        
        Args:
            file_path: 파일 경로
            encoding: 파일 인코딩 (기본값: utf-8)
        
        Returns:
            List[str]: 추가된 문서의 ID 리스트
        """
        logger.info(f"파일 로드 중: {file_path}")
        
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        
        # 청킹 및 추가
        chunks = self.split_text(content)
        
        # 메타데이터에 소스 파일 정보 추가
        metadatas = [{"source": file_path} for _ in chunks]
        
        return self.add_texts(texts=chunks, metadatas=metadatas)
    
    def clear(self):
        """
        Vector Store의 모든 문서를 삭제합니다.
        
        주의: 이 작업은 되돌릴 수 없습니다.
        """
        logger.warning("Vector Store의 모든 문서를 삭제합니다.")
        
        # 새 Vector Store 생성으로 초기화
        self._vector_store = self._create_vector_store()
        
        logger.info("Vector Store가 초기화되었습니다.")


# 테스트용 코드
if __name__ == "__main__":
    # 간단한 테스트
    print("VectorStoreManager 테스트")
    
    # 💡 get_embeddings()는 내부적으로 .env의 EMBEDDING_PROVIDER 설정을 따릅니다.
    # 특정 provider를 강제하고 싶다면 get_embeddings(provider="openai") 처럼 사용 가능합니다.
    try:
        from utils.llm_factory import get_embeddings
        # 기본 임베딩 모델 로드 (OpenAI 또는 Ollama)
        embeddings = get_embeddings()
        
        manager = VectorStoreManager(embeddings=embeddings, collection_name="test_collection")
        
        # 테스트 문서 추가
        test_texts = [
            "LangGraph는 LangChain 위에 구축된 상태 기반 에이전트 프레임워크입니다.",
            "RAG는 Retrieval-Augmented Generation의 약자로, 검색 증강 생성을 의미합니다.",
            "Vector Store는 임베딩 벡터를 저장하고 유사도 검색을 수행합니다.",
        ]
        
        print("\n📥 테스트 데이터 추가 중...")
        manager.add_texts(test_texts)
        
        # 검색 테스트
        query = "LangGraph란 무엇인가요?"
        results = manager.search(query, k=2)
        
        print(f"\n🔍 검색 테스트 (쿼리: '{query}')")
        print(f"결과 ({len(results)}개):")
        for i, doc in enumerate(results, 1):
            print(f"   [{i}] {doc.page_content[:100]}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
