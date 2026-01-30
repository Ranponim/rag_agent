# -*- coding: utf-8 -*-
"""
Skill Loader Module
Parses SKILL.md files and converts them into LangChain Tools or Prompts.
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

@dataclass
class Skill:
    name: str
    description: str
    content: str
    path: Path
    metadata: Dict[str, Any]

class SkillLoader:
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        # Ensure absolute path relative to repo root if needed
        if not self.skills_dir.is_absolute():
            # Assuming this is run from repo root
            self.skills_dir = Path(os.getcwd()) / skills_dir

    def _parse_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        Parses YAML-like frontmatter manually without external dependencies.
        Returns (metadata_dict, remaining_content).
        """
        if not content.startswith("---"):
            return {}, content

        try:
            # Find the second '---'
            parts = content.split("---", 2)
            if len(parts) < 3:
                return {}, content

            frontmatter_str = parts[1]
            body = parts[2].strip()

            metadata = {}
            for line in frontmatter_str.strip().split('\n'):
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, val = line.split(':', 1)
                metadata[key.strip()] = val.strip()

            return metadata, body
        except Exception:
            return {}, content

    def load_skill_from_dir(self, skill_dir: Path) -> Optional[Skill]:
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return None

        content = skill_file.read_text(encoding="utf-8")
        metadata, body = self._parse_frontmatter(content)

        name = metadata.get("name", skill_dir.name)
        description = metadata.get("description", body[:100].replace('\n', ' '))

        return Skill(
            name=name,
            description=description,
            content=body,
            path=skill_dir,
            metadata=metadata
        )

    def load_all_skills(self) -> List[Skill]:
        skills = []
        if not self.skills_dir.exists():
            print(f"Warning: Skills directory {self.skills_dir} does not exist.")
            return []

        for item in self.skills_dir.iterdir():
            if item.is_dir():
                skill = self.load_skill_from_dir(item)
                if skill:
                    skills.append(skill)
        return skills

    def create_tool_from_skill(self, skill: Skill) -> StructuredTool:
        """
        Creates a LangChain StructuredTool that executes the skill's instructions.
        For simplicity in this 01e example, we assume the skill instruction
        contains a python command to run.
        """

        # Simple heuristic: extract the python command from the body
        # Look for `python path/to/script.py`
        # We ensure it ends with .py to avoid matching "python script:" text
        command_match = re.search(r'python\s+([^\s\n]+\.py)', skill.content)
        script_path = None
        if command_match:
             # Resolve path relative to repo root
             possible_path = command_match.group(1)
             if os.path.exists(possible_path):
                 script_path = possible_path
             else:
                 # Try relative to skill directory
                 possible_path_rel = skill.path / possible_path
                 if possible_path_rel.exists():
                     script_path = str(possible_path_rel)

        class SkillInput(BaseModel):
            query: Optional[str] = Field(default=None, description="Optional arguments for the skill. Ignored if skill takes no arguments.")

        def run_skill(query: Optional[str] = None) -> str:
            """Executes the skill script."""
            if not script_path:
                return f"Error: Could not find executable script in skill {skill.name}"

            try:
                # We ignore the query for this specific static script,
                # but a real implementation might pass arguments.
                cmd = [sys.executable, script_path]
                if query:
                     # If the script accepted args, we would append them here
                     # For now, just run the script as is
                     pass

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode != 0:
                    return f"Error executing skill:\n{result.stderr}"
                return result.stdout
            except Exception as e:
                return f"Execution failed: {str(e)}"

        return StructuredTool.from_function(
            func=run_skill,
            name=skill.name,
            description=skill.description,
            args_schema=SkillInput
        )
