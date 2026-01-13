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
    # =========================================================================
    # OpenAI 설정
    # =========================================================================
    openai_api_key: str = Field(
        default="1234qwer",
        description="OpenAI API 키 (또는 호환 서버의 인증 키)"
    )
    openai_model: str = Field(
        default="gpt-oss-20b",
        description="사용할 OpenAI 채팅 모델"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="사용할 OpenAI 임베딩 모델"
    )
    openai_api_base: str = Field(
        default="http://10.251.204.93:10000/v1",
        description="OpenAI 호환 API의 Base URL"
    )
    openai_embedding_api_base: Optional[str] = Field(
        default=None,
        description="OpenAI 호환 임베딩 API의 Base URL (생략 시 openai_api_base 사용)"
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
    logger.info(f"OpenAI API Base: {settings.openai_api_base}")
    if settings.openai_embedding_api_base:
        logger.info(f"OpenAI Embedding API Base: {settings.openai_embedding_api_base}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    
    return settings


# 모듈 로드 시 설정 로그 출력 (디버깅용)
if __name__ == "__main__":
    settings = get_settings()
    print(f"OpenAI Model: {settings.openai_model}")
    print(f"OpenAI API Base: {settings.openai_api_base}")
