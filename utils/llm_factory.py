# -*- coding: utf-8 -*-
"""
LLM 팩토리 모듈

이 모듈은 다양한 LLM 제공자(OpenAI, Ollama 등)를 위한 
팩토리 패턴을 구현합니다. 설정에 따라 적절한 LLM 인스턴스를 생성합니다.

주요 기능:
    - OpenAI ChatGPT 모델 생성 (Local LLM 호환)
    - 임베딩 모델 생성
    - 싱글톤 캐싱 지원
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    LLM 인스턴스를 생성하는 팩토리 클래스
    """
    
    _llm_cache: Optional[BaseChatModel] = None
    _embeddings_cache: dict[str, Embeddings] = {}
    
    @classmethod
    def create_openai_llm(
        cls,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        base_url: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """OpenAI ChatGPT 모델 인스턴스 생성"""
        
        # 캐시 확인 (kwargs가 없을 때만 캐시 사용)
        if not kwargs and cls._llm_cache is not None:
            return cls._llm_cache

        logger.info(f"LLM 생성 중... (모델: {model}, URL: {base_url})")
        
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            api_key=api_key or "dummy-key",
            model=model,
            temperature=temperature,
            base_url=base_url,
            **kwargs
        )
        
        # 캐시 저장 (kwargs가 없을 때만)
        if not kwargs:
            cls._llm_cache = llm

        return llm
    
    @classmethod
    def create_openai_embeddings(
        cls,
        api_key: str,
        model: str = "text-embedding-3-small",
        **kwargs
    ) -> Embeddings:
        """OpenAI 임베딩 모델 인스턴스 생성"""
        
        # 캐시 키 생성
        cache_key = f"openai_{model}"
        
        if not kwargs and cache_key in cls._embeddings_cache:
            return cls._embeddings_cache[cache_key]

        logger.info(f"OpenAI 임베딩 모델 생성 중... (모델: {model})")
        
        from langchain_openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            api_key=api_key or "dummy-key",
            model=model,
            base_url=kwargs.pop("base_url", None),
            **kwargs
        )
        
        if not kwargs:
            cls._embeddings_cache[cache_key] = embeddings

        return embeddings
    
    @classmethod
    def create_ollama_embeddings(
        cls,
        model: str = "nomic-embed-text",
        base_url: Optional[str] = None,
        **kwargs
    ) -> Embeddings:
        """Ollama 임베딩 모델 인스턴스 생성
        
        Args:
            model: Ollama 임베딩 모델명 (예: nomic-embed-text, mxbai-embed-large)
            base_url: Ollama 서버 URL (기본값: http://localhost:11434)
            **kwargs: 추가 파라미터
            
        Returns:
            Embeddings: Ollama 임베딩 인스턴스
        """
        
        # 캐시 키 생성
        cache_key = f"ollama_{model}"
        
        # 캐시 확인 (kwargs가 없을 때만 캐시 사용)
        if not kwargs and cache_key in cls._embeddings_cache:
            return cls._embeddings_cache[cache_key]

        logger.info(f"Ollama 임베딩 모델 생성 중... (모델: {model}, URL: {base_url})")
        
        from langchain_ollama import OllamaEmbeddings
        
        # OllamaEmbeddings 인스턴스 생성
        # base_url이 None이면 OllamaEmbeddings의 기본값(http://localhost:11434) 사용
        embeddings = OllamaEmbeddings(
            model=model,
            base_url=base_url,
            **kwargs
        )
        
        # 캐시 저장 (kwargs가 없을 때만)
        if not kwargs:
            cls._embeddings_cache[cache_key] = embeddings

        return embeddings


def get_llm(**kwargs) -> BaseChatModel:
    """LLM 인스턴스 반환 (싱글톤)"""
    # 환경변수에서 LLM 설정 로드
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    model = os.getenv("OPENAI_MODEL", "local-model")
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    
    return LLMFactory.create_openai_llm(
        api_key=api_key,
        model=model,
        base_url=api_base,
        **kwargs
    )


def get_embeddings(**kwargs) -> Embeddings:
    """임베딩 인스턴스 반환 (싱글톤)
    
    환경변수 EMBEDDING_PROVIDER에 따라 OpenAI 또는 Ollama 임베딩을 반환합니다.
    
    Args:
        **kwargs: 임베딩 모델 생성 시 전달할 추가 파라미터
        
    Returns:
        Embeddings: 설정된 provider에 맞는 임베딩 인스턴스
        
    Example:
        >>> embeddings = get_embeddings()  # EMBEDDING_PROVIDER에 따라 자동 선택
        >>> vectors = embeddings.embed_documents(["Hello", "World"])
    """
    # 인자로 넘어온 provider를 우선하고 없으면 환경변수 사용
    provider = kwargs.pop("provider", os.getenv("EMBEDDING_PROVIDER", "openai")).lower()
    
    if provider == "ollama":
        # Ollama 임베딩 사용 (로컬 기본 설정 사용)
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        
        logger.info(f"Ollama 임베딩 provider 사용 (모델: {model}, 로컬 기본 설정)")
        return LLMFactory.create_ollama_embeddings(
            model=model,
            **kwargs
        )
    else:
        # OpenAI 임베딩 사용 (기본값)
        api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "local-embedding-model")
        embedding_base = os.getenv("OPENAI_EMBEDDING_API_BASE") or os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
        
        logger.info(f"OpenAI 임베딩 provider 사용 (모델: {model})")
        return LLMFactory.create_openai_embeddings(
            api_key=api_key,
            model=model,
            base_url=embedding_base,
            **kwargs
        )


def log_llm_error(e: Exception):
    """LLM 관련 오류 상세 로깅
    
    Args:
        e: 발생한 예외 객체
        
    에러 타입별로 상세한 진단 정보를 제공하여 디버깅을 돕습니다.
    """
    import traceback
    
    # 동적 임포트 (모듈이 없을 수도 있으므로 try-except로 보호)
    try:
        import openai
    except ImportError:
        openai = None
    
    try:
        import httpx
    except ImportError:
        httpx = None
    
    # =========================================================================
    # 1. 기본 오류 정보 출력
    # =========================================================================
    error_type = type(e).__name__
    error_module = type(e).__module__
    
    print("\n" + "="*80)
    print("❌ LLM 오류 발생!")
    print("="*80)
    print(f"📌 오류 타입: {error_module}.{error_type}")
    print(f"📌 오류 메시지: {str(e)}")
    print("-"*80)
    
    # =========================================================================
    # 2. 오류 타입별 상세 진단
    # =========================================================================
    
    # httpx 연결 오류
    if httpx and isinstance(e, httpx.ConnectError):
        print("💡 진단: 서버 연결 실패")
        print("   원인:")
        print("   - LLM 서버가 실행되지 않았을 가능성")
        print("   - 방화벽이나 네트워크 문제")
        print("   - 잘못된 URL 설정")
        print("\n   해결 방법:")
        print("   1. Ollama를 사용 중이라면: 'ollama serve' 명령으로 서버 시작")
        print("   2. LM Studio를 사용 중이라면: LM Studio 앱에서 서버 시작")
        print("   3. URL 확인: .env 파일의 OPENAI_API_BASE 설정 확인")
    
    # httpx 타임아웃 오류
    elif httpx and isinstance(e, httpx.TimeoutException):
        print("💡 진단: 서버 응답 시간 초과")
        print("   원인:")
        print("   - 서버가 과부하 상태")
        print("   - 모델 로딩에 시간이 오래 걸림")
        print("   - 네트워크 지연")
        print("\n   해결 방법:")
        print("   1. 서버 상태 확인")
        print("   2. 더 가벼운 모델로 변경")
        print("   3. timeout 설정 증가")
    
    # httpx RemoteProtocolError
    elif httpx and isinstance(e, httpx.RemoteProtocolError):
        print("💡 진단: 서버 프로토콜 오류 (응답 중단)")
        print("   원인:")
        print("   - 서버가 예상치 못하게 연결을 끊음")
        print("   - 요청 형식이 서버와 맞지 않음")
        print("   - 서버 내부 오류")
        print("\n   해결 방법:")
        print("   1. 서버 로그 확인 (Ollama: 터미널 출력, LM Studio: 앱 로그)")
        print("   2. 서버 재시작")
        print("   3. 모델이 정상적으로 로드되었는지 확인")
        print("      - Ollama: 'ollama list' 명령으로 모델 확인")
        print("      - LM Studio: 로드된 모델 확인")
    
    # OpenAI API 상태 오류
    elif openai and isinstance(e, openai.APIStatusError):
        status_code = getattr(e, 'status_code', 'Unknown')
        print(f"💡 진단: API 상태 오류 (HTTP {status_code})")
        print("   원인:")
        if status_code == 404:
            print("   - 모델을 찾을 수 없음 (모델명이 잘못되었을 가능성)")
        elif status_code == 401:
            print("   - 인증 실패 (API 키가 잘못되었을 가능성)")
        elif status_code == 500:
            print("   - 서버 내부 오류")
        else:
            print(f"   - HTTP {status_code} 오류")
        print("\n   해결 방법:")
        print("   1. .env 파일의 OPENAI_MODEL 설정 확인")
        print("   2. 서버에서 사용 가능한 모델 목록 확인")
    
    # OpenAI API 연결 오류
    elif openai and isinstance(e, openai.APIConnectionError):
        print("💡 진단: API 연결 오류")
        print("   원인:")
        print("   - API 서버에 연결할 수 없음")
        print("   - 네트워크 문제")
        print("\n   해결 방법:")
        print("   1. 인터넷 연결 확인")
        print("   2. 프록시 설정 확인")
        print("   3. 방화벽 설정 확인")
    
    # 기타 오류
    else:
        print("💡 진단: 알 수 없는 오류 타입")
        print(f"   오류 객체 타입: {type(e)}")
        print(f"   상세 메시지: {str(e)}")
    
    # =========================================================================
    # 3. 환경 변수 설정 상태 출력
    # =========================================================================
    print("\n" + "-"*80)
    print("🔍 현재 환경 변수 설정:")
    print("-"*80)
    
    env_vars = {
        "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
        "OPENAI_API_KEY": "****" + (os.getenv("OPENAI_API_KEY", "")[-4:] if os.getenv("OPENAI_API_KEY") else "없음"),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
        "EMBEDDING_PROVIDER": os.getenv("EMBEDDING_PROVIDER"),
        "OLLAMA_EMBEDDING_MODEL": os.getenv("OLLAMA_EMBEDDING_MODEL"),
    }
    
    for key, value in env_vars.items():
        print(f"   {key}: {value or '(설정되지 않음)'}")
    
    # =========================================================================
    # 4. 스택 트레이스 출력
    # =========================================================================
    print("\n" + "-"*80)
    print("📋 전체 스택 트레이스:")
    print("-"*80)
    traceback.print_exc()
    
    # =========================================================================
    # 5. 연결 테스트 방법 안내
    # =========================================================================
    print("\n" + "-"*80)
    print("🔧 연결 테스트 방법:")
    print("-"*80)
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:11434")
    
    if "11434" in api_base:
        print("   Ollama 서버 테스트:")
        print(f"   curl {api_base.replace('/v1', '')}")
        print("   (정상이면 'Ollama is running' 메시지 표시)")
    elif "1234" in api_base:
        print("   LM Studio 서버 테스트:")
        print(f"   curl {api_base}/models")
        print("   (정상이면 사용 가능한 모델 목록 JSON 반환)")
    else:
        print(f"   API 서버 테스트:")
        print(f"   curl {api_base}")
    
    print("="*80 + "\n")
