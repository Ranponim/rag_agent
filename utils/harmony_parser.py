# -*- coding: utf-8 -*-
"""
Harmony 포맷 파서 모듈

GPT-OSS 모델의 Harmony 포맷 응답을 LangChain 표준 형식으로 변환합니다.

문제:
    GPT-OSS는 tool call을 AIMessage.content에 JSON 문자열로 반환하고,
    tool_calls 속성은 빈 리스트로 유지합니다.

해결:
    content의 JSON을 파싱하여 tool_calls 속성을 채웁니다.
"""

import json
import logging
import uuid
from typing import List, Any, Optional

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def parse_harmony_tool_call(
    response: AIMessage,
    available_tools: List[BaseTool]
) -> AIMessage:
    """
    GPT-OSS Harmony 포맷 응답을 파싱하여 tool_calls 속성을 채웁니다.
    
    Args:
        response: LLM으로부터 받은 AIMessage
        available_tools: 사용 가능한 도구 리스트
        
    Returns:
        tool_calls가 채워진 AIMessage (수정이 필요한 경우)
        또는 원본 response (수정이 필요 없는 경우)
    """
    # 이미 tool_calls가 있으면 그대로 반환
    if response.tool_calls:
        logger.debug("tool_calls가 이미 존재함 - 파싱 스킵")
        return response
    
    # content가 없거나 문자열이 아니면 그대로 반환
    if not response.content or not isinstance(response.content, str):
        return response
    
    content = response.content.strip()
    
    # JSON이 아닌 일반 텍스트면 그대로 반환
    if not (content.startswith('{') or content.startswith('[')):
        return response
    
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.debug("content가 유효한 JSON이 아님 - 파싱 스킵")
        return response
    
    # 도구 파라미터와 매칭 시도
    matched_tool = _match_tool(parsed, available_tools)
    
    if matched_tool is None:
        logger.debug("매칭되는 도구 없음 - 파싱 스킵")
        return response
    
    tool_name, tool_args = matched_tool
    
    # 새로운 tool_calls 생성
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "name": tool_name,
        "args": tool_args
    }
    
    logger.info(f"🔧 Harmony tool call 파싱 성공: {tool_name}({tool_args})")
    
    # 새 AIMessage 생성
    # additional_kwargs에 들어있는 "refusal": null 등이 vLLM 서버에서 오류를 일으킬 수 있으므로 비움
    return AIMessage(
        content="", 
        tool_calls=[tool_call],
        # additional_kwargs는 비워서 전송 시 문제를 방지함
        additional_kwargs={}, 
        id=response.id
    )


def _match_tool(
    parsed_json: dict,
    available_tools: List[BaseTool]
) -> Optional[tuple]:
    """
    파싱된 JSON이 어떤 도구의 파라미터와 매칭되는지 확인합니다.
    
    Args:
        parsed_json: 파싱된 JSON 객체
        available_tools: 사용 가능한 도구 리스트
        
    Returns:
        (tool_name, tool_args) 튜플 또는 None
    """
    if not isinstance(parsed_json, dict):
        return None
    
    for tool in available_tools:
        # 도구의 스키마에서 파라미터 이름 추출
        schema = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        # JSON 키가 도구 파라미터와 일치하는지 확인
        json_keys = set(parsed_json.keys())
        
        # required 파라미터가 모두 있고, JSON 키가 properties에 포함되면 매칭
        if required and required.issubset(json_keys) and json_keys.issubset(set(properties.keys())):
            logger.debug(f"도구 매칭: {tool.name}")
            return (tool.name, parsed_json)
        
        # required가 없는 경우, 키가 하나라도 일치하면 매칭 (단순 휴리스틱)
        if not required and json_keys and json_keys.issubset(set(properties.keys())):
            logger.debug(f"도구 매칭 (휴리스틱): {tool.name}")
            return (tool.name, parsed_json)
    
    return None
