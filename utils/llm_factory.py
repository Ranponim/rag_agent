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
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    LLM 인스턴스를 생성하는 팩토리 클래스
    """
    
    _llm_cache: Optional[BaseChatModel] = None
    _embeddings_cache: Optional[Embeddings] = None
    
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
        
        if not kwargs and cls._embeddings_cache is not None:
            return cls._embeddings_cache

        logger.info(f"임베딩 모델 생성 중... (모델: {model})")
        
        from langchain_openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            api_key=api_key or "dummy-key",
            model=model,
            base_url=kwargs.pop("base_url", None),
            **kwargs
        )
        
        if not kwargs:
            cls._embeddings_cache = embeddings

        return embeddings


def get_llm(**kwargs) -> BaseChatModel:
    """LLM 인스턴스 반환 (싱글톤)"""
    from config.settings import get_settings
    settings = get_settings()
    
    return LLMFactory.create_openai_llm(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_api_base,
        **kwargs
    )


def get_embeddings(**kwargs) -> Embeddings:
    """임베딩 인스턴스 반환 (싱글톤)"""
    from config.settings import get_settings
    settings = get_settings()
    
    embedding_base_url = settings.openai_embedding_api_base or settings.openai_api_base
    
    return LLMFactory.create_openai_embeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        base_url=embedding_base_url,
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
