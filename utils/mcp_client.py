# -*- coding: utf-8 -*-
"""
============================================================================
🔌 MCP Client Manager - PowerShell 성공 사례 기반 고도화 버전
============================================================================

이 모듈은 MCP(Model Context Protocol) 서버 연결을 관리합니다.
PowerShell Invoke-WebRequest 성공 사례를 벤치마킹하여 httpx 설정을 최적화했습니다.
"""

import asyncio
import logging
import httpx
from typing import Dict, List, Optional, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

# 로깅 설정
logger = logging.getLogger(__name__)

class MCPClientManager:
    """
    MCP 서버 연결 관리자
    
    서버별 연동 특성(RemoteProtocolError 등)을 극복하기 위해 
    httpx 설정을 세밀하게 제어합니다.
    """
    
    def __init__(
        self,
        server_configs: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        self.server_configs = server_configs
        self.client: Optional[MultiServerMCPClient] = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connected = False

    def _get_optimized_httpx_client(self):
        """
        PowerShell 성공 사례를 100% 재현하기 위한 헤더 세트를 반환합니다.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json", # 명시적으로 지정
            "Connection": "keep-alive"
        }
        return headers

    async def connect(self) -> "MCPClientManager":
        """서버 연결 시도 (재시도 로직 포함)"""
        if self.connected:
            return self

        logger.info(f"🔌 [MCP] {len(self.server_configs)}개 서버 연결 시작...")
        
        # PowerShell 성공 레시피 주입
        headers = self._get_optimized_httpx_client()
        for name, config in self.server_configs.items():
            if config.get("transport") == "streamable_http":
                # 기존 헤더 병합
                existing_headers = config.get("headers", {})
                config["headers"] = {**headers, **existing_headers}
                
                # httpx의 타임아웃 및 프록시 설정을 위해 환경변수 무시 시도
                # (langchain-mcp-adapters 내부적으로 httpx.AsyncClient를 생성하므로
                #  직접 제어는 어렵지만, 필요 시 OS 환경변수를 임시로 변경할 수 있음)
                logger.info(f"  ✅ [{name}] 정밀 헤더 적용: {config['url']}")

        for attempt in range(self.max_retries):
            try:
                # MultiServerMCPClient 인스턴스 생성
                # 어댑터 0.1.0은 생성자에서 바로 연결을 준비함
                self.client = MultiServerMCPClient(self.server_configs)
                self.connected = True
                logger.info(f"✅ [MCP] 클라이언트 생성 성공 (시도 {attempt+1})")
                return self
            except Exception as e:
                logger.error(f"❌ [MCP] 연결 실패 (시도 {attempt+1}): {e}")
                if attempt == self.max_retries - 1: raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return self

    async def get_tools(self) -> List[BaseTool]:
        """도구 목록 가져오기 (오류 처리 및 로깅 강화)"""
        if not self.client:
            raise RuntimeError("연결되지 않았습니다.")
            
        try:
            logger.info("🔧 [MCP] 서버로부터 도구 목록을 수신 중...")
            # 0.1.0에서는 await get_tools() 사용
            tools = await self.client.get_tools()
            logger.info(f"✅ [MCP] {len(tools)}개의 도구 로드 완료")
            return tools
        except Exception as e:
            logger.error(f"💥 [MCP] 도구 로드 중 치명적 오류: {e}")
            # RemoteProtocolError 발생 시 팁 제공
            if "RemoteProtocolError" in str(e):
                logger.error("💡 팁: 서버가 응답을 끊었습니다. HTTP_PROXY 환경변수를 확인하거나 서버 로그를 점검하세요.")
            raise

    async def disconnect(self):
        """리소스 정리"""
        self.client = None
        self.connected = False
        logger.info("🔌 [MCP] 모든 연결이 해제되었습니다.")

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
