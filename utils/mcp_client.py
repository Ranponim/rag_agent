# -*- coding: utf-8 -*-
"""
============================================================================
🔌 MCP Client Manager - MCP 서버 연결 관리 유틸리티
============================================================================

이 모듈은 MCP(Model Context Protocol) 서버 연결을 관리하는 클래스를 제공합니다.
여러 MCP 서버와의 연결을 쉽게 관리하고, 오류 처리 및 재시도 로직을 포함합니다.

🎯 주요 기능:
    - 다중 MCP 서버 연결 관리
    - 자동 재시도 및 오류 복구
    - 상세한 로깅 (연결 상태, 도구 목록 등)
    - 안전한 리소스 정리

💡 SOLID 원칙:
    - Single Responsibility: MCP 연결 관리만 담당
    - Open/Closed: 새로운 transport 추가 시 확장 가능
    - Dependency Injection: 서버 설정을 외부에서 주입
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool


# =============================================================================
# 📝 로깅 설정
# =============================================================================

# 로거 생성 - 이 모듈의 모든 로그를 관리
logger = logging.getLogger(__name__)


# =============================================================================
# 🔌 MCP Client Manager 클래스
# =============================================================================

class MCPClientManager:
    """
    MCP 서버 연결을 관리하는 클래스
    
    이 클래스는 MultiServerMCPClient를 래핑하여 더 편리한 인터페이스를 제공합니다.
    연결 재시도, 오류 처리, 상세한 로깅 등의 기능을 추가합니다.
    
    Attributes:
        server_configs (Dict[str, Any]): MCP 서버 설정 딕셔너리
        client (Optional[MultiServerMCPClient]): 실제 MCP 클라이언트 인스턴스
        max_retries (int): 연결 재시도 최대 횟수
        retry_delay (float): 재시도 간 기본 대기 시간(초)
        connected (bool): 연결 상태 플래그
    
    Example:
        >>> manager = MCPClientManager({
        ...     "analysis_llm": {
        ...         "transport": "streamable_http",
        ...         "url": "http://165.213.69.30:8001/mcp"
        ...     }
        ... })
        >>> await manager.connect()
        >>> tools = await manager.get_tools()
        >>> await manager.disconnect()
    """
    
    def __init__(
        self,
        server_configs: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        MCP Client Manager 초기화
        
        Args:
            server_configs: MCP 서버 설정 딕셔너리
                각 서버는 다음 형식을 따라야 합니다:
                {
                    "server_name": {
                        "transport": "stdio" | "sse" | "streamable_http",
                        "url": "http://...",  # HTTP 기반일 때
                        "command": "python",  # stdio일 때
                        "args": [...],        # stdio일 때
                        "headers": {...}      # 선택사항: HTTP 헤더
                    }
                }
            max_retries: 연결 실패 시 재시도 최대 횟수 (기본: 3)
            retry_delay: 재시도 간 기본 대기 시간(초) (기본: 1.0)
                실제 대기 시간은 exponential backoff 적용: delay * (2 ** attempt)
        """
        # 서버 설정 저장 (외부에서 주입받음 - Dependency Injection)
        self.server_configs = server_configs
        
        # MCP 클라이언트 인스턴스 (아직 연결 전이므로 None)
        self.client: Optional[MultiServerMCPClient] = None
        
        # 재시도 설정
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 연결 상태 플래그
        self.connected = False
        
        # 서버 설정 검증 (초기화 시점에 오류 발견)
        self._validate_configs()
        
        # 초기화 완료 로그
        logger.info(f"🔧 [MCP] MCPClientManager 초기화 완료: {len(server_configs)}개 서버 설정")
    
    def _validate_configs(self) -> None:
        """
        서버 설정의 유효성을 검증합니다.
        
        필수 필드가 누락되었거나 잘못된 값이 있으면 예외를 발생시킵니다.
        이를 통해 런타임 오류를 사전에 방지합니다.
        
        Raises:
            ValueError: 설정이 유효하지 않을 때
        """
        # 서버 설정이 비어있으면 오류
        if not self.server_configs:
            raise ValueError("❌ MCP 서버 설정이 비어있습니다. 최소 1개 이상의 서버가 필요합니다.")
        
        # 각 서버별로 설정 검증
        for server_name, config in self.server_configs.items():
            # transport 필드는 필수
            if "transport" not in config:
                raise ValueError(
                    f"❌ 서버 '{server_name}'의 'transport' 필드가 누락되었습니다."
                )
            
            # transport 값 검증
            transport = config["transport"]
            valid_transports = ["stdio", "sse", "streamable_http"]
            if transport not in valid_transports:
                raise ValueError(
                    f"❌ 서버 '{server_name}'의 transport '{transport}'가 유효하지 않습니다. "
                    f"사용 가능한 값: {valid_transports}"
                )
            
            # HTTP 기반 transport는 url 필수
            if transport in ["sse", "streamable_http"]:
                if "url" not in config:
                    raise ValueError(
                        f"❌ 서버 '{server_name}'의 transport가 '{transport}'인데 'url' 필드가 누락되었습니다."
                    )
            
            # stdio transport는 command와 args 필수
            if transport == "stdio":
                if "command" not in config:
                    raise ValueError(
                        f"❌ 서버 '{server_name}'의 transport가 'stdio'인데 'command' 필드가 누락되었습니다."
                    )
                if "args" not in config:
                    raise ValueError(
                        f"❌ 서버 '{server_name}'의 transport가 'stdio'인데 'args' 필드가 누락되었습니다."
                    )
        
        # 모든 검증 통과
        logger.debug(f"✅ [MCP] 서버 설정 검증 완료")
    
    async def connect(self) -> "MCPClientManager":
        """
        MCP 서버에 연결합니다.
        
        연결 실패 시 자동으로 재시도하며, exponential backoff를 적용합니다.
        모든 재시도가 실패하면 예외를 발생시킵니다.
        
        Returns:
            MCPClientManager: self를 반환하여 메서드 체이닝 가능
        
        Raises:
            Exception: 모든 재시도가 실패했을 때
        
        Example:
            >>> manager = await MCPClientManager(configs).connect()
        """
        # 이미 연결되어 있으면 재연결하지 않음
        if self.connected and self.client:
            logger.warning("⚠️ [MCP] 이미 연결되어 있습니다. 기존 연결을 유지합니다.")
            return self
        
        # 재시도 루프
        for attempt in range(self.max_retries):
            try:
                # 시도 번호 로그 (1부터 시작)
                logger.info(f"🔌 [MCP] 연결 시도 {attempt + 1}/{self.max_retries}")
                
                # 각 서버별 연결 정보 출력
                for server_name, config in self.server_configs.items():
                    transport = config["transport"]
                    logger.info(f"  📡 [{server_name}] Transport: {transport}")
                    
                    # transport 종류에 따라 다른 정보 출력
                    if transport in ["sse", "streamable_http"]:
                        logger.info(f"  🌐 [{server_name}] URL: {config['url']}")
                    elif transport == "stdio":
                        logger.info(f"  💻 [{server_name}] Command: {config['command']} {config['args']}")
                
                # MultiServerMCPClient 생성 및 연결
                self.client = MultiServerMCPClient(self.server_configs)
                
                # async context manager 진입 (__aenter__ 호출)
                # 이 단계에서 실제 서버 연결이 수행됩니다.
                await self.client.__aenter__()
                
                # 연결 성공
                self.connected = True
                logger.info(f"✅ [MCP] 모든 서버 연결 성공!")
                
                return self
                
            except Exception as e:
                # 연결 실패 로그
                logger.error(f"❌ [MCP] 연결 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                
                # 마지막 시도였으면 예외 발생
                if attempt == self.max_retries - 1:
                    logger.error(f"💥 [MCP] 모든 재시도 실패. 연결을 포기합니다.")
                    raise
                
                # 재시도 전 대기 (exponential backoff)
                wait_time = self.retry_delay * (2 ** attempt)
                logger.info(f"⏳ [MCP] {wait_time}초 후 재시도합니다...")
                await asyncio.sleep(wait_time)
        
        # 이 지점에는 도달하지 않아야 함 (위에서 예외 발생)
        raise RuntimeError("❌ [MCP] 예기치 않은 연결 오류")
    
    async def get_tools(self) -> List[BaseTool]:
        """
        연결된 모든 MCP 서버에서 사용 가능한 도구를 가져옵니다.
        
        Returns:
            List[BaseTool]: LangChain Tool 객체 리스트
        
        Raises:
            RuntimeError: 연결되지 않은 상태에서 호출했을 때
            Exception: 도구 가져오기 실패 시
        
        Example:
            >>> tools = await manager.get_tools()
            >>> print([t.name for t in tools])
            ['analyze_3gpp', 'search_spec', ...]
        """
        # 연결 상태 확인
        if not self.connected or not self.client:
            error_msg = "❌ [MCP] 서버에 연결되지 않았습니다. connect()를 먼저 호출하세요."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # MCP 클라이언트에서 도구 가져오기
            logger.info("🔧 [MCP] 도구 목록 가져오는 중...")
            tools = self.client.get_tools()
            
            # 도구 정보 로그
            logger.info(f"✅ [MCP] {len(tools)}개의 도구를 발견했습니다:")
            for tool in tools:
                # 각 도구의 이름과 설명 출력
                logger.info(f"  🔨 {tool.name}: {tool.description}")
            
            return tools
            
        except Exception as e:
            # 도구 가져오기 실패
            logger.error(f"❌ [MCP] 도구 가져오기 실패: {e}")
            raise
    
    async def disconnect(self) -> None:
        """
        MCP 서버 연결을 안전하게 종료합니다.
        
        리소스를 정리하고 연결 상태를 초기화합니다.
        이미 연결이 끊어진 상태에서 호출해도 안전합니다.
        
        Example:
            >>> await manager.disconnect()
        """
        # 연결되지 않은 상태면 종료할 것이 없음
        if not self.connected or not self.client:
            logger.debug("ℹ️ [MCP] 이미 연결이 종료되어 있습니다.")
            return
        
        try:
            # async context manager 종료 (__aexit__ 호출)
            logger.info("🔌 [MCP] 연결 종료 중...")
            await self.client.__aexit__(None, None, None)
            
            # 상태 초기화
            self.client = None
            self.connected = False
            
            logger.info("✅ [MCP] 연결이 안전하게 종료되었습니다.")
            
        except Exception as e:
            # 종료 중 오류 발생 (로그만 남기고 예외는 발생시키지 않음)
            logger.error(f"⚠️ [MCP] 연결 종료 중 오류 발생: {e}")
            
            # 상태는 어쨌든 초기화
            self.client = None
            self.connected = False
    
    async def __aenter__(self) -> "MCPClientManager":
        """
        Async context manager 진입 (with 문 사용 시)
        
        자동으로 connect()를 호출합니다.
        
        Returns:
            MCPClientManager: 연결된 매니저 인스턴스
        
        Example:
            >>> async with MCPClientManager(configs) as manager:
            ...     tools = await manager.get_tools()
        """
        # connect()를 호출하고 self 반환
        return await self.connect()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Async context manager 종료 (with 문 종료 시)
        
        자동으로 disconnect()를 호출합니다.
        
        Args:
            exc_type: 예외 타입 (예외가 없으면 None)
            exc_val: 예외 값
            exc_tb: 예외 traceback
        """
        # disconnect()를 호출하여 리소스 정리
        await self.disconnect()
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        현재 설정된 서버 정보를 반환합니다.
        
        디버깅이나 로깅 목적으로 사용됩니다.
        
        Returns:
            Dict[str, Any]: 서버 이름과 설정 정보
        
        Example:
            >>> info = manager.get_server_info()
            >>> print(info)
            {
                'analysis_llm': {
                    'transport': 'streamable_http',
                    'url': 'http://...'
                }
            }
        """
        return self.server_configs.copy()
    
    @property
    def is_connected(self) -> bool:
        """
        연결 상태를 확인하는 프로퍼티
        
        Returns:
            bool: 연결되어 있으면 True, 아니면 False
        
        Example:
            >>> if manager.is_connected:
            ...     print("연결됨")
        """
        return self.connected and self.client is not None
