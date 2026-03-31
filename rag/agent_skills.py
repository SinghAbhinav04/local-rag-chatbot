"""
Agent Skills — loads skill definitions from SKILL.md files.

Priority:  workspace ./skills/  >  global ~/.rag-agent/skills/
Each skill folder must contain a SKILL.md with optional YAML frontmatter.
Bundled skills from rag/bundled_skills/ are auto-installed to global dir on first run.
"""

import os
import re
import shutil

GLOBAL_SKILLS_DIR = os.path.expanduser("~/.rag-agent/skills")
WORKSPACE_SKILLS_DIR = os.path.join(os.getcwd(), "skills")
BUNDLED_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "bundled_skills")


def _auto_install_bundled():
    """Copy bundled skills to global dir if not already present."""
    if not os.path.isdir(BUNDLED_SKILLS_DIR):
        return
    os.makedirs(GLOBAL_SKILLS_DIR, exist_ok=True)
    for entry in os.listdir(BUNDLED_SKILLS_DIR):
        src = os.path.join(BUNDLED_SKILLS_DIR, entry)
        dst = os.path.join(GLOBAL_SKILLS_DIR, entry)
        if os.path.isdir(src) and not os.path.isdir(dst):
            shutil.copytree(src, dst)


# Auto-install on import
_auto_install_bundled()


def _parse_frontmatter(content: str):
    """Extract YAML frontmatter and body from a markdown file."""
    if not content.startswith("---"):
        return {}, content
    lines = content.split("\n")
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx == -1:
        return {}, content

    frontmatter_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:]).lstrip("\n")

    # Simple YAML key: value parser (no dependency needed)
    meta = {}
    for line in frontmatter_lines:
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta, body


def list_skills() -> list:
    """List all available skills from workspace and global dirs."""
    skills = []
    seen = set()

    for source, skills_dir in [("workspace", WORKSPACE_SKILLS_DIR), ("global", GLOBAL_SKILLS_DIR)]:
        if not os.path.isdir(skills_dir):
            continue
        for entry in sorted(os.listdir(skills_dir)):
            skill_dir = os.path.join(skills_dir, entry)
            skill_file = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isdir(skill_dir) or not os.path.isfile(skill_file):
                continue

            with open(skill_file, "r") as f:
                content = f.read()

            meta, body = _parse_frontmatter(content)
            name = meta.get("name", entry)
            description = meta.get("description", "")

            # Extract description from first paragraph if not in frontmatter
            if not description:
                for line in body.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        description = line[:200]
                        break

            if name in seen:
                continue
            seen.add(name)

            skills.append({
                "name": name,
                "description": description,
                "source": source,
                "path": skill_file,
            })

    return skills


def load_skill(name: str) -> str | None:
    """Load a skill's content by name (frontmatter stripped). Returns None if not found."""
    for skills_dir in [WORKSPACE_SKILLS_DIR, GLOBAL_SKILLS_DIR]:
        # Try matching by folder name
        skill_file = os.path.join(skills_dir, name, "SKILL.md")
        if os.path.isfile(skill_file):
            with open(skill_file, "r") as f:
                content = f.read()
            _, body = _parse_frontmatter(content)
            return body

    # Try matching by metadata name
    for skill in list_skills():
        if skill["name"] == name:
            with open(skill["path"], "r") as f:
                content = f.read()
            _, body = _parse_frontmatter(content)
            return body

    return None


def build_skills_summary() -> str:
    """Build a summary of all available skills for the system prompt."""
    skills = list_skills()
    if not skills:
        return ""

    lines = ["Available Skills:"]
    for s in skills:
        lines.append(f"  - **{s['name']}** ({s['source']}): {s['description']}")
    return "\n".join(lines)
