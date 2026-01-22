# -*- coding: utf-8 -*-
"""
Harmony 포맷 파서 및 호환성 유틸리티

GPT-OSS 모델의 Harmony 포맷 응답을 LangChain 표준 형식으로 변환하고,
vLLM 서버와의 통신 시 메시지 호환성을 보장합니다.
"""

import json
import logging
import uuid
from typing import List, Any, Optional, Union

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, BaseMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def parse_harmony_tool_call(
    response: AIMessage,
    available_tools: List[BaseTool]
) -> AIMessage:
    """
    GPT-OSS Harmony 포맷 응답을 파싱하여 tool_calls 속성을 채웁니다.
    """
    if response.tool_calls:
        return response
    
    if not response.content or not isinstance(response.content, str):
        return response
    
    content = response.content.strip()
    if not (content.startswith('{') or content.startswith('[')):
        return response
    
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return response
    
    matched_tool = _match_tool(parsed, available_tools)
    if matched_tool is None:
        return response
    
    tool_name, tool_args = matched_tool
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "name": tool_name,
        "args": tool_args
    }
    
    logger.info(f"🔧 Harmony tool call 파싱 성공: {tool_name}({tool_args})")
    
    # 💡 중요: 원래 content(JSON 문자열)를 유지하면서 tool_calls만 추가합니다.
    # 이렇게 해야 나중에 LLM에 다시 보낼 때 원래 모습 그대로 보낼 수 있습니다.
    return AIMessage(
        content=response.content, 
        tool_calls=[tool_call],
        additional_kwargs={}, 
        id=response.id
    )


def clean_history_for_harmony(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    vLLM 서버(GPT-OSS)가 거부감을 느끼지 않도록 대화 기록을 엄격하게 정제합니다.
    """
    cleaned = []
    for msg in messages:
        # 1. tool_calls가 포함된 Assistant 메시지 처리
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # content가 비어있으면 서버가 400 에러를 낼 수 있음
            content = msg.content if msg.content else "Calling tool..."
            cleaned.append(AIMessage(content=content))
        
        # 2. Tool 역할의 메시지 처리 (vLLM이 싫어함)
        elif isinstance(msg, ToolMessage):
            # User 역할로 위장하여 전송하고 content가 문자열인지 확인
            content = str(msg.content) if msg.content else "No result."
            cleaned.append(HumanMessage(content=f"Observation: {content}"))
            
        elif isinstance(msg, HumanMessage):
             cleaned.append(HumanMessage(content=str(msg.content)))
             
        elif isinstance(msg, SystemMessage):
             cleaned.append(SystemMessage(content=str(msg.content)))
        
        else:
            cleaned.append(msg)
            
    return cleaned


def _match_tool(parsed_json: dict, available_tools: List[BaseTool]) -> Optional[tuple]:
    if not isinstance(parsed_json, dict):
        return None
    
    for tool in available_tools:
        schema = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        json_keys = set(parsed_json.keys())
        
        if required and required.issubset(json_keys) and json_keys.issubset(set(properties.keys())):
            return (tool.name, parsed_json)
        if not required and json_keys and json_keys.issubset(set(properties.keys())):
            return (tool.name, parsed_json)
    return None
