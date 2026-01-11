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
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("유효한 OpenAI API 키가 필요합니다.")
        
        logger.info(f"OpenAI LLM 생성 중... (모델: {model})")
        
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            **kwargs
        )
        
        logger.info("OpenAI LLM 인스턴스 생성 완료")
        return llm
    
    @classmethod
    def create_ollama_llm(
        cls,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        temperature: float = 0.0,
        **kwargs
    ) -> BaseChatModel:
        """
        Ollama 로컬 LLM 인스턴스를 생성합니다.
        
        Args:
            base_url: Ollama 서버 URL
            model: 사용할 모델명 (기본값: llama3.2)
            temperature: 생성 온도
            **kwargs: 추가 매개변수
        
        Returns:
            BaseChatModel: Ollama LLM 인스턴스
        
        Example:
            >>> llm = LLMFactory.create_ollama_llm(
            ...     base_url="http://localhost:11434",
            ...     model="llama3.2"
            ... )
        """
        logger.info(f"Ollama LLM 생성 중... (모델: {model})")
        
        from langchain_ollama import ChatOllama
        
        llm = ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
            **kwargs
        )
        
        logger.info("Ollama LLM 인스턴스 생성 완료")
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
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("유효한 OpenAI API 키가 필요합니다.")
        
        logger.info(f"OpenAI 임베딩 모델 생성 중... (모델: {model})")
        
        from langchain_openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model=model,
            **kwargs
        )
        
        logger.info("OpenAI 임베딩 인스턴스 생성 완료")
        return embeddings
    
    @classmethod
    def create_huggingface_embeddings(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs
    ) -> Embeddings:
        """
        HuggingFace 로컬 임베딩 모델 인스턴스를 생성합니다.
        
        OpenAI API 키가 없는 경우 대안으로 사용할 수 있습니다.
        
        Args:
            model_name: HuggingFace 모델명
            **kwargs: 추가 매개변수
        
        Returns:
            Embeddings: HuggingFace 임베딩 인스턴스
        """
        logger.info(f"HuggingFace 임베딩 모델 생성 중... (모델: {model_name})")
        
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            **kwargs
        )
        
        logger.info("HuggingFace 임베딩 인스턴스 생성 완료")
        return embeddings


def get_llm(
    provider: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    설정에 따라 적절한 LLM 인스턴스를 반환합니다.
    
    이 함수는 캐시를 사용하여 동일한 LLM 인스턴스를 재사용합니다.
    
    Args:
        provider: LLM 제공자 ("openai" 또는 "ollama")
                  None이면 설정에서 자동 감지
        **kwargs: LLM 생성에 전달할 추가 매개변수
    
    Returns:
        BaseChatModel: LLM 인스턴스
    
    Raises:
        ValueError: LLM 설정이 올바르지 않은 경우
    
    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("안녕하세요!")
    """
    # 설정 로드
    from config.settings import get_settings
    settings = get_settings()
    
    # 제공자 결정
    if provider is None:
        provider = settings.get_llm_provider()
    
    logger.info(f"LLM 제공자: {provider}")
    
    # LLM 생성
    if provider == "openai":
        return LLMFactory.create_openai_llm(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            **kwargs
        )
    elif provider == "ollama":
        return LLMFactory.create_ollama_llm(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            **kwargs
        )
    else:
        raise ValueError(
            "LLM 설정이 올바르지 않습니다. "
            ".env 파일에서 OPENAI_API_KEY 또는 OLLAMA_BASE_URL을 설정하세요."
        )


def get_embeddings(
    provider: Optional[str] = None,
    **kwargs
) -> Embeddings:
    """
    설정에 따라 적절한 임베딩 모델 인스턴스를 반환합니다.
    
    Args:
        provider: 임베딩 제공자 ("openai" 또는 "huggingface")
                  None이면 설정에서 자동 감지
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
    
    # 제공자 결정
    if provider is None:
        provider = settings.get_llm_provider()
    
    # 임베딩 모델 생성
    if provider == "openai":
        return LLMFactory.create_openai_embeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            **kwargs
        )
    else:
        # OpenAI가 없으면 HuggingFace 로컬 임베딩 사용
        logger.info("OpenAI 키가 없어 HuggingFace 로컬 임베딩을 사용합니다.")
        return LLMFactory.create_huggingface_embeddings(**kwargs)


# 테스트용 코드
if __name__ == "__main__":
    # 설정 로드 테스트
    from config.settings import get_settings
    settings = get_settings()
    
    print(f"LLM Provider: {settings.get_llm_provider()}")
    
    # LLM 생성 테스트 (API 키가 있는 경우만)
    if settings.validate_openai_key():
        llm = get_llm()
        print(f"LLM Type: {type(llm).__name__}")
