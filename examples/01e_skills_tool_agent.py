# -*- coding: utf-8 -*-
"""
============================================================================
📚 01e. Skills Tool Agent - Claude Skills as Tools
============================================================================

이 예제는 Claude Style의 Skills(SKILL.md)를 로드하여 LangGraph 에이전트의
도구(Tool)로 활용하는 방법을 보여줍니다.

🎯 학습 목표:
    1. SkillLoader를 통한 SKILL.md 파일 파싱
    2. 파싱된 스킬을 LangChain StructuredTool로 변환
    3. ReAct 에이전트에서 로컬 스킬 실행

💡 핵심 개념:
    - Skills: Claude Code 스타일의 도구 정의 (YAML frontmatter + Markdown)
    - Wrapper Approach: 스킬을 LangChain Tool로 래핑하여 에이전트에 주입

실행 방법:
    python examples/01e_skills_tool_agent.py
"""

import sys
import os
import asyncio
from pathlib import Path

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.skill_loader import SkillLoader

# =============================================================================
# 🤖 Agent 생성 함수
# =============================================================================

def create_skills_agent():
    """
    Skills 폴더에서 스킬을 로드하고 에이전트를 생성합니다.
    """
    # 1. LLM 초기화
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")

    model = ChatOpenAI(
        base_url=api_base,
        api_key=api_key,
        model=model_name
    )

    print(f"\n{'='*70}")
    print(f"🤖 [Agent] LLM 모델 초기화: {model_name}")
    print(f"🌐 [Agent] API Base: {api_base}")
    print(f"{'='*70}\n")

    # 2. Skill 로드 및 Tool 변환
    loader = SkillLoader(skills_dir="skills")
    skills = loader.load_all_skills()

    tools = []
    print(f"📦 [Skills] 로드된 스킬:")
    for skill in skills:
        tool = loader.create_tool_from_skill(skill)
        tools.append(tool)
        print(f"  - {skill.name}: {skill.description}")

    if not tools:
        print("⚠️ 경고: 로드된 스킬이 없습니다. 'skills/' 폴더를 확인하세요.")

    # 3. 에이전트 생성
    system_prompt = """당신은 로컬 시스템 관리 및 정보를 제공하는 유능한 AI 어시스턴트입니다.
제공된 도구(Skills)를 적극적으로 활용하여 사용자의 요청을 처리하세요.
모든 답변은 한국어로 작성해 주세요."""

    agent = create_react_agent(
        model,
        tools=tools,
        prompt=system_prompt
    )

    print(f"✅ [Agent] Skills Agent 생성 완료\n")
    return agent

# =============================================================================
# 🔄 대화형 실행 루프
# =============================================================================

async def run_interactive():
    print(f"\n{'='*70}")
    print("💬 Skills Agent Interactive Chat Mode")
    print(f"{'='*70}\n")

    app = create_skills_agent()
    chat_history = []

    print("\n✅ 준비 완료! 대화를 시작하세요. (종료하려면 'q' 입력)")
    print(f"{'-'*70}\n")

    while True:
        try:
            query = input("\n🙋 User: ").strip()
            if not query:
                continue
            if query.lower() in ['q', 'quit', 'exit']:
                print("\n👋 종료합니다.")
                break

            current_messages = chat_history + [HumanMessage(content=query)]
            print(f"\n🤖 Agent 생각 중...", end="", flush=True)

            final_response = None
            step_count = 0

            # 스트리밍 실행
            async for chunk in app.astream(
                {"messages": current_messages},
                stream_mode="values"
            ):
                if "messages" in chunk:
                    messages = chunk["messages"]
                    if messages:
                        last_msg = messages[-1]

                        # 도구 호출 로깅
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            step_count += 1
                            print(f"\n\n🔧 [Step {step_count}] 도구 호출:")
                            for tool_call in last_msg.tool_calls:
                                print(f"  📌 {tool_call.get('name')}: {tool_call.get('args')}")
                            print("  ⏳ 실행 중...", end="", flush=True)

                    final_response = chunk

            # 최종 응답 처리
            if final_response and "messages" in final_response:
                final_messages = final_response["messages"]
                last_msg = final_messages[-1]
                if hasattr(last_msg, 'content') and last_msg.content:
                    print(f"\n\n🤖 Agent:\n{last_msg.content}\n")
                chat_history = final_messages

        except KeyboardInterrupt:
            print("\n\n⚠️ 인터럽트. 종료합니다.")
            break
        except EOFError:
            print("\n\n👋 EOF 감지. 종료합니다.")
            break
        except Exception as e:
            print(f"\n\n❌ 오류: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(run_interactive())
    except KeyboardInterrupt:
        pass
