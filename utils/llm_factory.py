# -*- coding: utf-8 -*-
"""
LLM 팩토리 모듈

이 모듈은 다양한 LLM 제공자(OpenAI, Ollama 등)를 위한 
팩토리 패턴을 구현합니다. 설정에 따라 적절한 LLM 인스턴스를 생성합니다.

주요 기능:
    - OpenAI ChatGPT 모델 생성
    - Ollama 로컬 모델 생성 (선택)
    - 임베딩 모델 생성

사용 예시:
    from utils.llm_factory import get_llm, get_embeddings
    
    # LLM 인스턴스 생성
    llm = get_llm()
    
    # 임베딩 모델 생성
    embeddings = get_embeddings()
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    LLM 인스턴스를 생성하는 팩토리 클래스
    
    이 클래스는 설정에 따라 OpenAI 또는 Ollama LLM을 생성합니다.
    팩토리 패턴을 사용하여 LLM 생성 로직을 캡슐화합니다.
    
    Attributes:
        _llm_cache: 생성된 LLM 인스턴스 캐시 (싱글톤 패턴)
        _embeddings_cache: 생성된 임베딩 인스턴스 캐시
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
        """
        OpenAI ChatGPT 모델 인스턴스를 생성합니다.
        
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델명 (기본값: gpt-4o-mini)
            temperature: 생성 온도 (0.0 = 결정적, 1.0 = 창의적)
            **kwargs: 추가 매개변수
        
        Returns:
            BaseChatModel: OpenAI ChatGPT 인스턴스
        
        Raises:
            ValueError: API 키가 유효하지 않은 경우
        
        Example:
            >>> llm = LLMFactory.create_openai_llm(
            ...     api_key="sk-...",
            ...     model="gpt-4o-mini"
            ... )
        """
        if not api_key:
             # 로컬 LLM을 위해 더미 키를 허용하지만 경고를 남길 수 있습니다.
             # 여기서는 호출자가 처리했다고 가정합니다.
             pass
        
        logger.info(f"OpenAI 호환 LLM 생성 중... (모델: {model}, URL: {base_url or 'default'})")
        
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            api_key=api_key or "dummy-key", # API key is required by library, use dummy if empty
            model=model,
            temperature=temperature,
            base_url=base_url,
            **kwargs
        )
        
        logger.info("OpenAI 호환 LLM 인스턴스 생성 완료")
        return llm
    
    @classmethod
    def create_openai_embeddings(
        cls,
        api_key: str,
        model: str = "text-embedding-3-small",
        **kwargs
    ) -> Embeddings:
        """
        OpenAI 임베딩 모델 인스턴스를 생성합니다.
        
        Args:
            api_key: OpenAI API 키
            model: 임베딩 모델명 (기본값: text-embedding-3-small)
            **kwargs: 추가 매개변수
        
        Returns:
            Embeddings: OpenAI 임베딩 인스턴스
        
        Example:
            >>> embeddings = LLMFactory.create_openai_embeddings(
            ...     api_key="sk-...",
            ...     model="text-embedding-3-small"
            ... )
        """
        # 로컬 LLM의 임베딩을 사용하는 경우 시작 문자열 검증 로직은 제거하거나 완화해야 합니다.
        
        logger.info(f"OpenAI 임베딩 모델 생성 중... (모델: {model})")
        
        from langchain_openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            api_key=api_key or "dummy-key",
            model=model,
            base_url=kwargs.pop("base_url", None),
            **kwargs
        )
        
        logger.info("OpenAI 임베딩 인스턴스 생성 완료")
        return embeddings


def get_llm(**kwargs) -> BaseChatModel:
    """
    설정에 따라 적절한 LLM 인스턴스를 반환합니다.
    
    이 함수는 캐시를 사용하여 동일한 LLM 인스턴스를 재사용합니다.
    
    Args:
        **kwargs: LLM 생성에 전달할 추가 매개변수
    
    Returns:
        BaseChatModel: LLM 인스턴스
    
    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("안녕하세요!")
    """
    # 설정 로드
    from config.settings import get_settings
    settings = get_settings()
    
    # LLM 생성 (무조건 OpenAI 호환 사용)
    return LLMFactory.create_openai_llm(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_api_base,
        **kwargs
    )


def get_embeddings(**kwargs) -> Embeddings:
    """
    설정에 따라 적절한 임베딩 모델 인스턴스를 반환합니다.
    
    Args:
        **kwargs: 임베딩 생성에 전달할 추가 매개변수
    
    Returns:
        Embeddings: 임베딩 인스턴스
    
    Example:
        >>> embeddings = get_embeddings()
        >>> vector = embeddings.embed_query("안녕하세요")
    """
    # 설정 로드
    from config.settings import get_settings
    settings = get_settings()
    
    # 임베딩 모델 생성 (무조건 OpenAI 호환 사용)
    # 임베딩 전용 URL이 있으면 사용, 없으면 기본 API Base URL 사용
    embedding_base_url = settings.openai_embedding_api_base or settings.openai_api_base
    
    return LLMFactory.create_openai_embeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        base_url=embedding_base_url,
        **kwargs
    )


def log_llm_error(e: Exception, llm: Optional[BaseChatModel] = None):
    """
    LLM 관련 오류 발생 시 상세한 정보를 로깅합니다.
    
    Args:
        e: 발생한 예외 객체
        llm: (선택) 관련 LLM 인스턴스 (URL 정보 추출용)
    """
    import httpx
    import openai
    
    error_type = type(e).__name__
    base_url = "unknown"
    if llm:
        base_url = getattr(llm, "openai_api_base", "unknown")
    
    logger.error(f"❌ LLM 오류 발생! (Type: {error_type})")
    if llm:
        logger.error(f"📍 Target URL: {base_url}")
    
    if isinstance(e, openai.APIConnectionError):
        logger.error(f"💡 원인: 서버에 연결할 수 없습니다. URL이 올바른지 확인하세요.")
        logger.error(f"👉 상세: {str(e)}")
    elif isinstance(e, httpx.ConnectError):
        logger.error(f"💡 원인: 네트워크 연결 거부됨. 서버가 실행 중인지 확인하세요.")
    elif isinstance(e, openai.AuthenticationError):
        logger.error(f"💡 원인: 인증 실패. API Key가 올바른지 확인하세요.")
    elif isinstance(e, openai.BadRequestError):
        logger.error(f"💡 원인: 잘못된 요청입니다. 모델명이나 파라미터를 확인하세요.")
        logger.error(f"👉 상세: {str(e)}")
    else:
        logger.error(f"⚠️ 기타 오류: {str(e)}")


# 테스트용 코드
if __name__ == "__main__":
    # 설정 로드 테스트
    from config.settings import get_settings
    settings = get_settings()
    
    print(f"LLM Base URL: {settings.openai_api_base}")
    
    llm = get_llm()
    print(f"LLM Type: {type(llm).__name__}")
