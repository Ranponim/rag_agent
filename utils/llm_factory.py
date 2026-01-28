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
    """LLM 관련 오류 상세 로깅"""
    import openai
    import httpx
    
    error_type = type(e).__name__
    logger.error(f"❌ LLM 오류 발생! (Type: {error_type})")
    
    if isinstance(e, httpx.ConnectError):
        logger.error(f"💡 원인: 서버 연결 실패. Local LLM이 켜져 있는지, URL이 올바른지 확인하세요.")
    elif isinstance(e, openai.APIStatusError):
        logger.error(f"💡 원인: API 상태 오류 ({e.status_code}). 모델명이나 서버 상태를 확인하세요.")
    else:
        logger.error(f"⚠️ 상세: {str(e)}")
