"""
Agent Tools — tool registry and built-in tools, inspired by PicoClaw.

Tools are callable by the LLM via Ollama's native tool-calling support.
Each tool has: name, description, parameters (JSON schema), execute().
"""

import os
import re
import json
import subprocess
import tempfile
import traceback

from rag import agent_memory

# ── Deny patterns for shell exec (ported from PicoClaw) ───────────────────────
DENY_PATTERNS = [
    re.compile(r'\brm\s+-[rf]{1,2}\b'),
    re.compile(r'\bsudo\b'),
    re.compile(r'\bshutdown\b'),
    re.compile(r'\breboot\b'),
    re.compile(r'\bpoweroff\b'),
    re.compile(r'\bmkfs\b'),
    re.compile(r'\bdd\s+if='),
    re.compile(r'\bchmod\s+[0-7]{3,4}\b'),
    re.compile(r'\bchown\b'),
    re.compile(r'\bpkill\b'),
    re.compile(r'\bkillall\b'),
    re.compile(r'\bkill\b'),
    re.compile(r':\(\)\s*\{.*\};\s*:'),
    re.compile(r'\beval\b'),
    re.compile(r'\bgit\s+push\b'),
    re.compile(r'\bgit\s+force\b'),
    re.compile(r'\bdocker\s+run\b'),
    re.compile(r'\bapt\s+(install|remove|purge)\b'),
    re.compile(r'\bpip\s+install\b'),
    re.compile(r'\bnpm\s+install\s+-g\b'),
]


class Tool:
    """Base class for all tools."""
    def name(self) -> str:
        raise NotImplementedError
    def description(self) -> str:
        raise NotImplementedError
    def parameters(self) -> dict:
        raise NotImplementedError
    def execute(self, args: dict) -> dict:
        raise NotImplementedError

    def to_ollama_schema(self) -> dict:
        """Convert to Ollama tool-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": self.parameters(),
            }
        }


class ToolRegistry:
    """Manages and executes registered tools."""
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name()] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, args: dict) -> dict:
        tool = self.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not found"}
        try:
            return tool.execute(args)
        except Exception as e:
            return {"error": f"Tool '{name}' failed: {str(e)}"}

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name(), "description": t.description(), "parameters": t.parameters()}
            for t in self._tools.values()
        ]

    def get_ollama_tools(self) -> list[dict]:
        """Get tool definitions in Ollama's format."""
        return [t.to_ollama_schema() for t in self._tools.values()]


# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN TOOLS
# ══════════════════════════════════════════════════════════════════════════════

class WebSearchTool(Tool):
    """Search the web using DuckDuckGo (no API key needed)."""
    def name(self): return "web_search"
    def description(self): return "Search the web using DuckDuckGo. Returns top results with titles, snippets, and URLs."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Max results (default 5)", "default": 5},
            },
            "required": ["query"],
        }

    def execute(self, args: dict) -> dict:
        query = args.get("query", "")
        max_results = int(args.get("max_results", 5))
        if not query:
            return {"error": "Missing query"}
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return {
                "results": [
                    {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                    for r in results
                ]
            }
        except ImportError:
            return {"error": "duckduckgo-search package not installed. Run: pip install duckduckgo-search"}
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}


class ShellExecTool(Tool):
    """Execute shell commands with safety guards."""
    def name(self): return "exec"
    def description(self): return "Execute a shell command and return its output. Dangerous commands are blocked for safety."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "working_dir": {"type": "string", "description": "Optional working directory"},
            },
            "required": ["command"],
        }

    def execute(self, args: dict) -> dict:
        command = args.get("command", "").strip()
        if not command:
            return {"error": "Missing command"}

        # Safety check
        lower = command.lower()
        for pattern in DENY_PATTERNS:
            if pattern.search(lower):
                return {"error": "Command blocked by safety guard (dangerous pattern detected)"}

        cwd = args.get("working_dir") or os.getcwd()

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=30, cwd=cwd,
            )
            output = result.stdout
            if result.stderr:
                output += "\nSTDERR:\n" + result.stderr
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
            if not output.strip():
                output = "(no output)"
            # Truncate
            if len(output) > 8000:
                output = output[:8000] + f"\n... (truncated, {len(output)-8000} more chars)"
            return {"output": output}
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out (30s limit)"}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}


class ReadFileTool(Tool):
    """Read file contents."""
    def name(self): return "read_file"
    def description(self): return "Read the contents of a file. Supports offset/length for large files."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {"type": "integer", "description": "Byte offset to start reading from (default 0)"},
                "length": {"type": "integer", "description": "Maximum bytes to read (default 64KB)"},
            },
            "required": ["path"],
        }

    def execute(self, args: dict) -> dict:
        path = args.get("path", "")
        if not path:
            return {"error": "Missing path"}
        if not os.path.exists(path):
            return {"error": f"File not found: {path}"}

        offset = int(args.get("offset", 0))
        length = min(int(args.get("length", 65536)), 65536)

        try:
            with open(path, "r", errors="replace") as f:
                f.seek(offset)
                content = f.read(length)
            total_size = os.path.getsize(path)
            has_more = offset + len(content.encode()) < total_size
            return {
                "content": content,
                "file": os.path.basename(path),
                "total_bytes": total_size,
                "has_more": has_more,
            }
        except Exception as e:
            return {"error": f"Read failed: {str(e)}"}


class WriteFileTool(Tool):
    """Write content to a file."""
    def name(self): return "write_file"
    def description(self): return "Write content to a file. Creates parent directories if needed."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    def execute(self, args: dict) -> dict:
        path = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return {"error": "Missing path"}
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return {"status": "ok", "path": path, "bytes_written": len(content)}
        except Exception as e:
            return {"error": f"Write failed: {str(e)}"}


class ListDirTool(Tool):
    """List directory contents."""
    def name(self): return "list_dir"
    def description(self): return "List files and directories at a path."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list (default: current dir)"},
            },
            "required": [],
        }

    def execute(self, args: dict) -> dict:
        path = args.get("path", ".")
        if not os.path.isdir(path):
            return {"error": f"Not a directory: {path}"}
        try:
            entries = []
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                entry = {"name": name, "type": "DIR" if os.path.isdir(full) else "FILE"}
                if os.path.isfile(full):
                    entry["size"] = os.path.getsize(full)
                entries.append(entry)
            return {"entries": entries, "count": len(entries)}
        except Exception as e:
            return {"error": f"List failed: {str(e)}"}


class PythonExecTool(Tool):
    """Execute Python code in a sandboxed subprocess."""
    def name(self): return "python_exec"
    def description(self): return "Execute a Python code snippet and return its output. Runs in a separate process with a 15s timeout."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        }

    def execute(self, args: dict) -> dict:
        code = args.get("code", "")
        if not code:
            return {"error": "Missing code"}
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp_path = f.name
            try:
                result = subprocess.run(
                    ["python3", tmp_path],
                    capture_output=True, text=True, timeout=15,
                )
                output = result.stdout
                if result.stderr:
                    output += "\nSTDERR:\n" + result.stderr
                if not output.strip():
                    output = "(no output)"
                return {"output": output, "exit_code": result.returncode}
            finally:
                os.unlink(tmp_path)
        except subprocess.TimeoutExpired:
            return {"error": "Code execution timed out (15s limit)"}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}


class MemoryReadTool(Tool):
    """Read from agent memory."""
    def name(self): return "memory_read"
    def description(self): return "Read from the agent's persistent memory. Returns long-term memory and recent daily notes."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Which section: 'long_term', 'today', or 'all' (default: all)",
                    "enum": ["long_term", "today", "all"],
                },
            },
            "required": [],
        }

    def execute(self, args: dict) -> dict:
        section = args.get("section", "all")
        result = {}
        if section in ("long_term", "all"):
            result["long_term"] = agent_memory.read_long_term()
        if section in ("today", "all"):
            result["today"] = agent_memory.read_today()
        if section == "all":
            result["recent_notes"] = agent_memory.get_recent_notes(3)
        return result


class MemoryWriteTool(Tool):
    """Write to agent memory."""
    def name(self): return "memory_write"
    def description(self): return "Write to the agent's persistent memory. Use this to remember important facts, preferences, or notes."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "'long_term' to update core memory, 'today' to add daily note",
                    "enum": ["long_term", "today"],
                },
                "content": {"type": "string", "description": "The content to write/append"},
                "mode": {
                    "type": "string",
                    "description": "'append' to add to existing content, 'overwrite' to replace (long_term only)",
                    "enum": ["append", "overwrite"],
                },
            },
            "required": ["section", "content"],
        }

    def execute(self, args: dict) -> dict:
        section = args.get("section", "")
        content = args.get("content", "")
        mode = args.get("mode", "append")

        if not content:
            return {"error": "Missing content"}

        if section == "long_term":
            if mode == "overwrite":
                agent_memory.write_long_term(content)
            else:
                agent_memory.append_long_term(content)
            return {"status": "ok", "section": "long_term"}
        elif section == "today":
            agent_memory.append_today(content)
            return {"status": "ok", "section": "today"}
        else:
            return {"error": "Invalid section. Use 'long_term' or 'today'."}


class EditFileTool(Tool):
    """Edit a file by replacing old_text with new_text (ported from PicoClaw edit.go)."""
    def name(self): return "edit_file"
    def description(self): return "Edit a file by replacing old_text with new_text. The old_text must exist exactly once in the file."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to edit"},
                "old_text": {"type": "string", "description": "The exact text to find and replace"},
                "new_text": {"type": "string", "description": "The text to replace with"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    def execute(self, args: dict) -> dict:
        path = args.get("path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        if not path:
            return {"error": "Missing path"}
        if not os.path.isfile(path):
            return {"error": f"File not found: {path}"}

        try:
            with open(path, "r") as f:
                content = f.read()

            if old_text not in content:
                return {"error": "old_text not found in file. Make sure it matches exactly."}

            count = content.count(old_text)
            if count > 1:
                return {"error": f"old_text appears {count} times. Provide more context to make it unique."}

            new_content = content.replace(old_text, new_text, 1)
            with open(path, "w") as f:
                f.write(new_content)
            return {"status": "ok", "path": path}
        except Exception as e:
            return {"error": f"Edit failed: {str(e)}"}


class AppendFileTool(Tool):
    """Append content to the end of a file (ported from PicoClaw edit.go)."""
    def name(self): return "append_file"
    def description(self): return "Append content to the end of a file. Creates the file if it doesn't exist."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The file path to append to"},
                "content": {"type": "string", "description": "The content to append"},
            },
            "required": ["path", "content"],
        }

    def execute(self, args: dict) -> dict:
        path = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return {"error": "Missing path"}
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a") as f:
                f.write(content)
            return {"status": "ok", "path": path}
        except Exception as e:
            return {"error": f"Append failed: {str(e)}"}


class WebFetchTool(Tool):
    """Fetch a URL and return text content (ported from PicoClaw web.go WebFetchTool)."""
    def name(self): return "web_fetch"
    def description(self): return "Fetch a URL and return its text content. Strips HTML to readable text. Max 50000 chars."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "max_chars": {"type": "integer", "description": "Max characters to return (default 50000)"},
            },
            "required": ["url"],
        }

    def execute(self, args: dict) -> dict:
        url = args.get("url", "")
        max_chars = int(args.get("max_chars", 50000))
        if not url:
            return {"error": "Missing url"}
        try:
            import urllib.request
            import urllib.error
            headers = {"User-Agent": "Mozilla/5.0 (RAG Agent; +rag-system) AppleWebKit/537.36"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")

            # Strip HTML tags
            text = re.sub(r'<script[\s\S]*?</script>', '', raw)
            text = re.sub(r'<style[\s\S]*?</style>', '', text)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'[^\S\n]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

            if len(text) > max_chars:
                text = text[:max_chars] + f"\n... (truncated, {len(text)-max_chars} more chars)"
            return {"content": text, "url": url, "length": len(text)}
        except Exception as e:
            return {"error": f"Fetch failed: {str(e)}"}


class CronTool(Tool):
    """Schedule reminders and recurring tasks (ported from PicoClaw cron.go)."""
    # In-process scheduler using threading.Timer
    _jobs = {}
    _next_id = 1

    def name(self): return "cron"
    def description(self): return "Schedule reminders and recurring tasks. Use 'at_seconds' for one-time (e.g., 600 = 10 minutes). Use 'every_seconds' for recurring. Actions: add, list, remove."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Action: add, list, remove", "enum": ["add", "list", "remove"]},
                "message": {"type": "string", "description": "The reminder message"},
                "at_seconds": {"type": "integer", "description": "One-time: seconds from now"},
                "every_seconds": {"type": "integer", "description": "Recurring: interval in seconds"},
                "job_id": {"type": "string", "description": "Job ID (for remove)"},
            },
            "required": ["action"],
        }

    def execute(self, args: dict) -> dict:
        import threading
        import time

        action = args.get("action", "")

        if action == "list":
            if not CronTool._jobs:
                return {"jobs": [], "message": "No scheduled jobs"}
            jobs = []
            for jid, info in CronTool._jobs.items():
                jobs.append({"id": jid, "message": info["message"], "type": info["type"],
                             "created": info["created"]})
            return {"jobs": jobs}

        elif action == "remove":
            job_id = args.get("job_id", "")
            if job_id in CronTool._jobs:
                job = CronTool._jobs.pop(job_id)
                if job.get("timer"):
                    job["timer"].cancel()
                return {"status": "ok", "removed": job_id}
            return {"error": f"Job '{job_id}' not found"}

        elif action == "add":
            message = args.get("message", "")
            if not message:
                return {"error": "message is required for add"}

            at_seconds = args.get("at_seconds")
            every_seconds = args.get("every_seconds")

            if at_seconds and int(at_seconds) > 0:
                at_seconds = int(at_seconds)
                job_id = f"cron-{CronTool._next_id}"
                CronTool._next_id += 1

                def fire():
                    CronTool._jobs.pop(job_id, None)
                    print(f"\n🔔 REMINDER: {message}\n")

                timer = threading.Timer(at_seconds, fire)
                timer.daemon = True
                timer.start()

                CronTool._jobs[job_id] = {
                    "message": message, "type": "one-time",
                    "timer": timer, "created": time.strftime("%H:%M:%S"),
                }
                return {"status": "ok", "job_id": job_id, "fires_in": f"{at_seconds}s"}

            elif every_seconds and int(every_seconds) > 0:
                every_seconds = int(every_seconds)
                job_id = f"cron-{CronTool._next_id}"
                CronTool._next_id += 1

                def repeat():
                    while job_id in CronTool._jobs:
                        print(f"\n🔔 RECURRING: {message}\n")
                        time.sleep(every_seconds)

                t = threading.Thread(target=repeat, daemon=True)
                t.start()

                CronTool._jobs[job_id] = {
                    "message": message, "type": f"every {every_seconds}s",
                    "thread": t, "created": time.strftime("%H:%M:%S"),
                }
                return {"status": "ok", "job_id": job_id, "interval": f"{every_seconds}s"}
            else:
                return {"error": "at_seconds or every_seconds required for add"}
        else:
            return {"error": f"Unknown action: {action}"}


class FindSkillsTool(Tool):
    """Search installed skills (ported from PicoClaw skills_search.go)."""
    def name(self): return "find_skills"
    def description(self): return "Search for available skills by keyword. Returns matching skills with descriptions."
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for skills"},
            },
            "required": ["query"],
        }

    def execute(self, args: dict) -> dict:
        from rag import agent_skills
        query = args.get("query", "").lower()
        if not query:
            return {"error": "Missing query"}
        skills = agent_skills.list_skills()
        matches = []
        for s in skills:
            score = 0
            name_lower = s["name"].lower()
            desc_lower = s["description"].lower()
            for word in query.split():
                if word in name_lower:
                    score += 2
                if word in desc_lower:
                    score += 1
            if score > 0:
                matches.append({**s, "score": score})
        matches.sort(key=lambda x: x["score"], reverse=True)
        return {"results": matches[:10], "total": len(matches)}


# ── Registry factory ──────────────────────────────────────────────────────────

def create_default_registry() -> ToolRegistry:
    """Create a registry with all built-in tools."""
    reg = ToolRegistry()
    reg.register(WebSearchTool())
    reg.register(ShellExecTool())
    reg.register(ReadFileTool())
    reg.register(WriteFileTool())
    reg.register(ListDirTool())
    reg.register(PythonExecTool())
    reg.register(MemoryReadTool())
    reg.register(MemoryWriteTool())
    reg.register(EditFileTool())
    reg.register(AppendFileTool())
    reg.register(WebFetchTool())
    reg.register(CronTool())
    reg.register(FindSkillsTool())
    return reg
