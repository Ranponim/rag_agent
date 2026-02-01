# -*- coding: utf-8 -*-
"""
============================================================================
📦 RAG 데이터 로더 모듈 - 공통 데이터 로딩 및 벡터화 유틸리티
============================================================================

RAG 예제 파일들에서 공통으로 사용하는 데이터 로딩, 벡터화, 임베딩 기능을 제공합니다.

주요 기능:
    - 다양한 파일 형식 지원 (TXT, MD, CSV, PDF, XLSX, JSON, JSONL)
    - Vector Store 영속화 (한 번 임베딩한 데이터 재사용)
    - ./rag 폴더 파일 변경 감지 (추가/수정 시 자동 재임베딩)

사용 예시:
    from utils.data_loader import get_rag_vector_store
    
    # Vector Store 가져오기 (자동으로 데이터 로딩 및 임베딩)
    vs = get_rag_vector_store(collection_name="naive_rag")
    
    # 검색
    results = vs.search("LangGraph란?", k=3)
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Type

from langchain_core.documents import Document
from langchain_core.document_loaders.base import BaseLoader
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    JSONLoader  # JSON 파일용 (일반 JSON 처리)
)

from utils.llm_factory import get_embeddings
from utils.vector_store import VectorStoreManager

# 로거 설정
logger = logging.getLogger(__name__)


# =============================================================================
# 📄 커스텀 로더: JSONLineLoader (사용자 기존 코드 기반)
# =============================================================================

class JSONLineLoader(BaseLoader):
    """
    JSONL(Line-delimited JSON) 파일을 한 줄씩 읽어서 Document로 변환하는 로더
    
    각 줄 전체를 문자열로 임베딩하여 모든 필드가 검색 대상이 됩니다.
    일반 JSON 파일도 한 줄에 하나의 JSON 객체가 있으면 처리 가능합니다.
    
    Args:
        file_path: JSONL 파일 경로
        encoding: 파일 인코딩 (기본값: utf-8)
    
    Example:
        >>> loader = JSONLineLoader("data.jsonl")
        >>> docs = loader.load()
        >>> print(len(docs))  # 각 줄이 하나의 Document
    """
    
    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        """JSONLineLoader 초기화"""
        self.file_path = file_path       # 로드할 파일 경로
        self.encoding = encoding          # 파일 인코딩 (한글 지원을 위해 utf-8 사용)
    
    def load(self) -> List[Document]:
        """
        JSONL 파일을 로드하여 Document 리스트로 반환
        
        Returns:
            List[Document]: 각 줄을 Document로 변환한 리스트
        """
        docs = []  # 결과를 담을 리스트
        
        try:
            # 파일을 읽기 모드로 열기
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                # 파일을 한 줄씩 순회
                for line in f:
                    # 빈 줄이 아닌 경우에만 처리
                    if line.strip():
                        # 줄 전체를 page_content로, 파일 경로를 metadata에 저장
                        docs.append(Document(
                            page_content=line,  # JSON 문자열 전체가 임베딩 대상
                            metadata={"source": self.file_path}  # 출처 정보
                        ))
        except Exception as e:
            # 오류 발생 시 로그 출력 (프로그램은 계속 진행)
            print(f"Error loading {self.file_path}: {e}")
        
        return docs


# =============================================================================
# 🔧 파일 변경 감지 유틸리티
# =============================================================================

def _get_folder_hash(folder_path: str, extensions: List[str]) -> str:
    """
    폴더 내 파일들의 해시값을 계산하여 변경 감지에 사용
    
    파일 목록과 각 파일의 수정 시간을 조합하여 해시를 생성합니다.
    해시가 다르면 파일이 추가/수정/삭제된 것입니다.
    
    Args:
        folder_path: 감시할 폴더 경로
        extensions: 감시할 파일 확장자 목록
    
    Returns:
        str: 폴더 상태를 나타내는 해시 문자열
    """
    folder = Path(folder_path)
    
    # 폴더가 없으면 빈 해시 반환
    if not folder.exists():
        return "empty"
    
    # 파일 정보를 담을 리스트
    file_info = []
    
    # 지정된 확장자의 파일들을 찾아서 정보 수집
    for ext in extensions:
        for file_path in folder.rglob(f"*{ext}"):
            if file_path.is_file():
                # 파일명과 수정 시간을 문자열로 저장
                mtime = os.path.getmtime(file_path)
                file_info.append(f"{file_path}:{mtime}")
    
    # 파일 정보를 정렬하여 일관된 해시 생성
    file_info.sort()
    
    # 파일이 없으면 빈 해시 반환
    if not file_info:
        return "no_files"
    
    # 문자열을 합쳐서 MD5 해시 생성
    combined = "\n".join(file_info)
    return hashlib.md5(combined.encode()).hexdigest()


def _get_hash_file_path(persist_directory: str) -> str:
    """해시 파일 경로 반환"""
    return os.path.join(persist_directory, ".folder_hash")


def _read_saved_hash(persist_directory: str) -> Optional[str]:
    """저장된 해시값 읽기"""
    hash_file = _get_hash_file_path(persist_directory)
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return f.read().strip()
    return None


def _save_hash(persist_directory: str, hash_value: str):
    """해시값 저장"""
    hash_file = _get_hash_file_path(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)
    with open(hash_file, 'w') as f:
        f.write(hash_value)


# =============================================================================
# 📂 RAG 데이터 로더 클래스
# =============================================================================

class RAGDataLoader:
    """
    RAG 시스템을 위한 통합 데이터 로더
    
    다양한 파일 형식을 지원하며, ./rag 폴더의 파일을 자동으로 로드합니다.
    Vector Store 영속화와 파일 변경 감지 기능을 제공합니다.
    
    Attributes:
        source_dir: 데이터 소스 폴더 경로 (기본값: ./rag)
        encoding: 파일 인코딩 (기본값: utf-8)
    
    Example:
        >>> loader = RAGDataLoader()
        >>> docs = loader.load_all()
        >>> print(f"로드된 문서 수: {len(docs)}")
    """
    
    # 지원하는 파일 확장자와 로더 매핑
    # DirectoryLoader에서 사용할 로더 클래스들
    LOADER_MAP: Dict[str, Type[BaseLoader]] = {
        ".txt": TextLoader,           # 일반 텍스트 파일
        ".md": TextLoader,            # 마크다운 파일
        ".csv": CSVLoader,            # CSV 파일
        ".pdf": PyPDFLoader,          # PDF 문서
        ".xlsx": UnstructuredExcelLoader,  # Excel 파일
        # JSON: JSONLoader 사용 (별도 처리)
        # JSONL: JSONLineLoader 사용 (별도 처리)
    }
    
    def __init__(
        self, 
        source_dir: str = "./rag",
        encoding: str = "utf-8"
    ):
        """
        RAGDataLoader 초기화
        
        Args:
            source_dir: 데이터 소스 폴더 경로
            encoding: 파일 인코딩
        """
        self.source_dir = source_dir  # 데이터를 읽어올 폴더
        self.encoding = encoding       # 파일 인코딩
        
        # 폴더가 없으면 생성
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
            print(f"📁 {source_dir} 폴더가 생성되었습니다. 파일을 넣어주세요.")
    
    def load_all(self) -> List[Document]:
        """
        소스 폴더의 모든 지원 파일을 로드
        
        Returns:
            List[Document]: 로드된 모든 문서 리스트
        """
        print(f"\n📥 [데이터 로더] {self.source_dir} 폴더에서 파일 로딩 중...")
        
        all_documents = []
        
        # 각 확장자별로 파일 로드
        for ext, loader_cls in self.LOADER_MAP.items():
            try:
                docs = self._load_by_extension(ext, loader_cls)
                if docs:
                    all_documents.extend(docs)
                    print(f"   → {ext} 파일 {len(docs)}개 로드 완료")
            except Exception as e:
                # 특정 확장자 로드 실패해도 계속 진행
                print(f"   ⚠️ {ext} 로더 경고: {str(e)[:50]}...")
        
        # JSON 파일 로드 (LangChain JSONLoader 사용)
        try:
            json_docs = self._load_json_files()
            if json_docs:
                all_documents.extend(json_docs)
                print(f"   → .json 파일 {len(json_docs)}개 로드 완료")
        except Exception as e:
            print(f"   ⚠️ .json 로더 경고: {str(e)[:50]}...")
        
        # JSONL 파일 로드 (커스텀 JSONLineLoader 사용 - 한 줄씩 처리)
        try:
            jsonl_docs = self._load_jsonl_files()
            if jsonl_docs:
                all_documents.extend(jsonl_docs)
                print(f"   → .jsonl 파일 {len(jsonl_docs)}개 로드 완료")
        except Exception as e:
            print(f"   ⚠️ .jsonl 로더 경고: {str(e)[:50]}...")
        
        if all_documents:
            print(f"✅ 총 {len(all_documents)}개의 문서가 로드되었습니다.")
        else:
            print(f"   ⚠️ {self.source_dir} 폴더에 지원되는 파일이 없습니다.")
        
        return all_documents
    
    def _load_by_extension(
        self, 
        extension: str, 
        loader_cls: Type[BaseLoader]
    ) -> List[Document]:
        """
        특정 확장자의 파일들을 로드
        
        Args:
            extension: 파일 확장자 (예: ".txt")
            loader_cls: 사용할 로더 클래스
        
        Returns:
            List[Document]: 해당 확장자 파일들의 문서 리스트
        """
        # DirectoryLoader 사용
        loader = DirectoryLoader(
            path=self.source_dir,
            glob=f"**/*{extension}",
            loader_cls=loader_cls,
            loader_kwargs={"encoding": self.encoding},
            use_multithreading=False,  # Windows 안정성
            silent_errors=True
        )
        
        return loader.load()
    
    def _load_json_files(self) -> List[Document]:
        """
        JSON 파일을 LangChain JSONLoader로 로드
        
        일반 JSON 파일을 전체 구조로 파싱하여 Document로 변환합니다.
        
        Returns:
            List[Document]: 로드된 문서 리스트
        """
        documents = []
        source_path = Path(self.source_dir)
        
        # .json 파일 찾기
        for file_path in source_path.rglob("*.json"):
            if file_path.is_file():
                try:
                    # JSONLoader: 전체 JSON 구조를 텍스트로 변환
                    loader = JSONLoader(
                        file_path=str(file_path),
                        jq_schema=".",  # 전체 JSON 객체 선택
                        text_content=False  # JSON을 문자열로 변환
                    )
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    # jq 관련 오류 시 텍스트로 읽기 fallback
                    try:
                        with open(file_path, 'r', encoding=self.encoding) as f:
                            content = f.read()
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": str(file_path)}
                        ))
                    except Exception as fallback_e:
                        print(f"   ⚠️ JSON 로드 실패 {file_path}: {fallback_e}")
        
        return documents
    
    def _load_jsonl_files(self) -> List[Document]:
        """
        JSONL 파일을 커스텀 JSONLineLoader로 로드 (한 줄씩 처리)
        
        각 줄을 별도의 Document로 변환하여 개별 임베딩합니다.
        
        Returns:
            List[Document]: 로드된 문서 리스트
        """
        documents = []
        source_path = Path(self.source_dir)
        
        # .jsonl 파일 찾기
        for file_path in source_path.rglob("*.jsonl"):
            if file_path.is_file():
                loader = JSONLineLoader(
                    file_path=str(file_path),
                    encoding=self.encoding
                )
                docs = loader.load()
                documents.extend(docs)
        
        return documents
    
    def get_supported_extensions(self) -> List[str]:
        """지원하는 파일 확장자 목록 반환"""
        return list(self.LOADER_MAP.keys()) + [".json", ".jsonl"]


# =============================================================================
# 🚀 편의 함수: Vector Store 초기화 및 데이터 로딩 통합
# =============================================================================

def get_rag_vector_store(
    collection_name: str = "rag_collection",
    source_dir: str = "./rag",
    persist_dir: str = "./vector_db",
    embedding_provider: Optional[str] = None,
    force_reload: bool = False
) -> VectorStoreManager:
    """
    RAG용 Vector Store를 초기화하고 데이터를 로드합니다.
    
    ./rag 폴더의 파일 변경을 감지하여 필요한 경우에만 재임베딩합니다.
    한 번 임베딩한 데이터는 persist_dir에 저장되어 재사용됩니다.
    
    Args:
        collection_name: Vector Store 컬렉션 이름
        source_dir: 데이터 소스 폴더 (기본값: ./rag)
        persist_dir: Vector Store 영속화 폴더 (기본값: ./vector_db)
        embedding_provider: 임베딩 provider ("openai" 또는 "ollama", None이면 환경변수 사용)
        force_reload: True면 파일 변경 여부와 관계없이 강제 재임베딩
    
    Returns:
        VectorStoreManager: 초기화된 Vector Store 매니저
    
    Example:
        >>> vs = get_rag_vector_store(collection_name="naive_rag")
        >>> results = vs.search("LangGraph란?", k=3)
    """
    # 컬렉션별 영속화 경로 설정
    collection_persist_dir = os.path.join(persist_dir, collection_name)
    
    # 임베딩 모델 가져오기
    if embedding_provider:
        embeddings = get_embeddings(provider=embedding_provider)
    else:
        embeddings = get_embeddings()
    
    # Vector Store 매니저 생성 (영속화 경로 지정)
    manager = VectorStoreManager(
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=collection_persist_dir
    )
    
    # 파일 변경 감지를 위한 해시 계산
    loader = RAGDataLoader(source_dir=source_dir)
    extensions = loader.get_supported_extensions()
    current_hash = _get_folder_hash(source_dir, extensions)
    saved_hash = _read_saved_hash(collection_persist_dir)
    
    # 재임베딩 필요 여부 판단
    need_reload = force_reload or (current_hash != saved_hash)
    
    if need_reload:
        if force_reload:
            print("🔄 강제 리로드가 요청되었습니다. 데이터를 다시 임베딩합니다...")
        else:
            print("🔍 ./rag 폴더에 변경이 감지되었습니다. 데이터를 다시 임베딩합니다...")
        
        # 기존 데이터 삭제 후 새로 로드
        manager.clear()
        
        # 데이터 로드
        documents = loader.load_all()
        
        if documents:
            # Vector Store에 문서 추가 (자동으로 청킹 및 임베딩)
            manager.add_documents(documents)
            print(f"✅ {len(documents)}개의 문서가 Vector Store에 저장되었습니다.")
        else:
            # 데이터가 없으면 기본 텍스트 추가
            print("   ⚠️ 로딩된 문서가 없습니다. 기본 테스트 데이터를 적재합니다.")
            manager.add_texts(["LangGraph와 RAG 예제 데이터입니다."])
        
        # 현재 해시 저장
        _save_hash(collection_persist_dir, current_hash)
    else:
        print(f"✅ ./rag 폴더에 변경이 없습니다. 기존 임베딩을 재사용합니다.")
        print(f"   (저장 위치: {collection_persist_dir})")
    
    return manager


# =============================================================================
# 🧪 테스트 코드
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📦 RAG 데이터 로더 테스트")
    print("=" * 60)
    
    # Vector Store 가져오기 (자동 감지 및 임베딩)
    vs = get_rag_vector_store(collection_name="test_collection")
    
    # 검색 테스트
    query = "LangGraph란?"
    results = vs.search(query, k=2)
    
    print(f"\n🔍 검색 테스트 (쿼리: '{query}')")
    print(f"결과 ({len(results)}개):")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   [{i}] {preview}...")
