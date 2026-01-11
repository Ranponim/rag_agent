# -*- coding: utf-8 -*-
"""
LangGraph RAG Agent 학습 프로젝트 설정 모듈

이 모듈은 환경 변수를 로드하고 프로젝트 전반에서 사용되는 설정을 관리합니다.
Pydantic Settings를 활용하여 타입 안전하고 검증된 설정을 제공합니다.

사용 예시:
    from config.settings import get_settings
    
    settings = get_settings()
    print(settings.openai_api_key)
"""

import os
import logging
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로거 설정
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    애플리케이션 설정 클래스
    
    환경 변수에서 설정값을 로드하며, 기본값과 유효성 검사를 제공합니다.
    
    Attributes:
        openai_api_key: OpenAI API 키 (필수)
        openai_model: 사용할 OpenAI 모델명
        openai_embedding_model: 임베딩 모델명
        ollama_base_url: Ollama 서버 URL (선택)
        ollama_model: Ollama 모델명 (선택)
        log_level: 로깅 레벨
    """
    
    # =========================================================================
    # OpenAI 설정
    # =========================================================================
    openai_api_key: str = Field(
        default="",
        description="OpenAI API 키"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="사용할 OpenAI 채팅 모델"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="사용할 OpenAI 임베딩 모델"
    )
    
    # =========================================================================
    # Ollama 설정 (선택)
    # =========================================================================
    ollama_base_url: Optional[str] = Field(
        default=None,
        description="Ollama 서버 URL"
    )
    ollama_model: Optional[str] = Field(
        default="llama3.2",
        description="Ollama 모델명"
    )
    
    # =========================================================================
    # LangSmith 설정 (선택)
    # =========================================================================
    langchain_tracing_v2: bool = Field(
        default=False,
        description="LangSmith 추적 활성화 여부"
    )
    langchain_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API 키"
    )
    langchain_project: str = Field(
        default="langgraph-rag-learning",
        description="LangSmith 프로젝트명"
    )
    
    # =========================================================================
    # 일반 설정
    # =========================================================================
    log_level: str = Field(
        default="INFO",
        description="로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    class Config:
        """Pydantic 설정"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # 환경 변수는 대소문자 구분 안함
    
    def validate_openai_key(self) -> bool:
        """
        OpenAI API 키가 유효한 형식인지 확인합니다.
        
        Returns:
            bool: API 키가 유효한 형식이면 True
        """
        if not self.openai_api_key:
            logger.warning("OpenAI API 키가 설정되지 않았습니다.")
            return False
        if not self.openai_api_key.startswith("sk-"):
            logger.warning("OpenAI API 키 형식이 올바르지 않습니다.")
            return False
        return True
    
    def get_llm_provider(self) -> str:
        """
        사용 가능한 LLM 제공자를 반환합니다.
        
        OpenAI 키가 있으면 'openai', Ollama URL이 있으면 'ollama',
        둘 다 없으면 'none'을 반환합니다.
        
        Returns:
            str: 'openai', 'ollama', 또는 'none'
        """
        if self.validate_openai_key():
            return "openai"
        elif self.ollama_base_url:
            return "ollama"
        return "none"


@lru_cache()
def get_settings() -> Settings:
    """
    캐시된 Settings 인스턴스를 반환합니다.
    
    lru_cache 데코레이터를 사용하여 설정 객체를 한 번만 생성하고
    이후에는 캐시된 인스턴스를 반환합니다.
    
    Returns:
        Settings: 설정 인스턴스
    
    Example:
        >>> settings = get_settings()
        >>> print(settings.openai_model)
        gpt-4o-mini
    """
    logger.info("설정을 로드합니다...")
    settings = Settings()
    
    # 로깅 레벨 설정
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 설정 확인 로그
    provider = settings.get_llm_provider()
    logger.info(f"LLM 제공자: {provider}")
    
    if provider == "openai":
        logger.info(f"OpenAI 모델: {settings.openai_model}")
    elif provider == "ollama":
        logger.info(f"Ollama 모델: {settings.ollama_model}")
    else:
        logger.warning("LLM 제공자가 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    return settings


# 모듈 로드 시 설정 로그 출력 (디버깅용)
if __name__ == "__main__":
    settings = get_settings()
    print(f"OpenAI Model: {settings.openai_model}")
    print(f"LLM Provider: {settings.get_llm_provider()}")
