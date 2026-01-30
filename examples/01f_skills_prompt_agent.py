# -*- coding: utf-8 -*-
"""
============================================================================
📚 01f. Skills Prompt Agent - Claude Skills as Prompts
============================================================================

이 예제는 Claude Style의 Skills(SKILL.md)를 로드하여
시스템 프롬프트(System Prompt)에 주입하고, 에이전트가 이를 읽고
적절한 쉘 명령을 실행하도록 유도하는 방식을 보여줍니다.

🎯 학습 목표:
    1. SkillLoader를 사용하여 SKILL.md 본문 추출
    2. 시스템 프롬프트에 스킬 지침(Context) 주입 (Prompt Engineering)
    3. Generic ShellTool을 활용한 동적 스크립트 실행

💡 핵심 개념:
    - Native Approach: 스킬을 별도의 Tool로 감싸지 않고, 프롬프트로 지시사항을 전달
    - Dynamic Execution: 에이전트가 프롬프트를 읽고 스스로 명령어를 구성하여 실행

실행 방법:
    python examples/01f_skills_prompt_agent.py
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
from langchain_community.tools import ShellTool
from utils.skill_loader import SkillLoader

# =============================================================================
# 🤖 Agent 생성 함수
# =============================================================================

def create_prompt_skills_agent():
    """
    Skills 폴더에서 스킬을 로드하여 프롬프트에 추가하고 에이전트를 생성합니다.
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
    print(f"{'='*70}\n")

    # 2. Skill 로드 및 Prompt 구성
    loader = SkillLoader(skills_dir="skills")
    skills = loader.load_all_skills()

    skill_prompts = []
    print(f"📦 [Skills] 로드된 스킬 (Prompt Injection):")
    for skill in skills:
        # SKILL.md의 본문을 프롬프트에 추가
        formatted_skill = f"""
### Skill: {skill.name}
Description: {skill.description}
Instructions:
{skill.content}
"""
        skill_prompts.append(formatted_skill)
        print(f"  - {skill.name}")

    if not skill_prompts:
        print("⚠️ 경고: 로드된 스킬이 없습니다.")

    # 3. 도구 설정 (Generic Shell Tool)
    # 보안상 ShellTool은 위험할 수 있으므로 주의해서 사용해야 합니다.
    shell_tool = ShellTool()
    shell_tool.description = "Executes a shell command. Use this to run scripts mentioned in the skills."

    tools = [shell_tool]

    # 4. 시스템 프롬프트 구성
    base_system_prompt = """당신은 로컬 시스템 관리 및 정보를 제공하는 유능한 AI 어시스턴트입니다.
아래에 정의된 [Skills] 섹션을 참고하여 사용자의 요청을 처리하세요.
각 Skill에는 실행해야 할 명령어나 스크립트가 명시되어 있습니다.
제공된 'terminal' 도구를 사용하여 이 명령어를 실행하세요.
모든 답변은 한국어로 작성해 주세요.
"""

    full_prompt = base_system_prompt + "\n\n[Skills Available]\n" + "\n".join(skill_prompts)

    # 5. 에이전트 생성
    agent = create_react_agent(
        model,
        tools=tools,
        prompt=full_prompt
    )

    print(f"✅ [Agent] Prompt Skills Agent 생성 완료\n")
    return agent

# =============================================================================
# 🔄 대화형 실행 루프
# =============================================================================

async def run_interactive():
    print(f"\n{'='*70}")
    print("💬 Skills Prompt Agent Interactive Chat Mode")
    print(f"{'='*70}\n")

    app = create_prompt_skills_agent()
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

            async for chunk in app.astream(
                {"messages": current_messages},
                stream_mode="values"
            ):
                if "messages" in chunk:
                    messages = chunk["messages"]
                    if messages:
                        last_msg = messages[-1]

                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            step_count += 1
                            print(f"\n\n🔧 [Step {step_count}] 도구 호출:")
                            for tool_call in last_msg.tool_calls:
                                print(f"  📌 {tool_call.get('name')}: {tool_call.get('args')}")
                            print("  ⏳ 실행 중...", end="", flush=True)

                    final_response = chunk

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
