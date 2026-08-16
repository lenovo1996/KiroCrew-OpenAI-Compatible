"""mcp_handler_registry.py — Registry-based MCP tool handler system.

Local handlers for tools that must run in the OpenAI provider's Python process
(file I/O, bash, web_fetch, ask_question, spawn_run, spawn_list).

MCP-discovered tools (cron_add, cron_list, learn_add, etc.) are NOT registered
here — they are discovered from managed servers via in-process _list_tools()
and executed via MCP stdio protocol in mcp_executor.py.

Dispatch order (set by mcp_executor.py):
    1. Local registry lookup (O(1) dict, zero-latency)
    2. MCP stdio protocol (spawn server → tools/call → parse result)
    3. Error with available-tools hint
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Base Handler ─────────────────────────────────────────────────────────────

class McpHandler(ABC):
    """Base class for MCP tool handlers. Subclass → implement 3 methods → register."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def definition(self) -> dict:
        """OpenAI function-calling tool definition."""

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> str:
        """Execute and return string result. Prefix with '[Error' on failure."""


# ── Registry ─────────────────────────────────────────────────────────────────

class McpHandlerRegistry:
    """Central dispatch for local handlers. Thread-safe singleton via class methods."""

    _handlers: dict[str, McpHandler] = {}

    @classmethod
    def register(cls, handler: McpHandler) -> None:
        cls._handlers[handler.name] = handler

    @classmethod
    def register_many(cls, *handlers: McpHandler) -> None:
        for h in handlers:
            cls.register(h)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._handlers.keys())

    @classmethod
    def get(cls, name: str) -> McpHandler | None:
        return cls._handlers.get(name)

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._handlers

    @classmethod
    def definition_for(cls, name: str) -> dict | None:
        h = cls._handlers.get(name)
        return h.definition() if h else None

    @classmethod
    def all_definitions(cls) -> list[dict]:
        return [h.definition() for h in cls._handlers.values()]


# ── Built-in File/Bash/Web Handlers ──────────────────────────────────────────

class BashHandler(McpHandler):
    @property
    def name(self) -> str: return "execute_bash"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "execute_bash", "description": "Run a shell command and return its output.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "__tool_use_purpose": {"type": "string"}}, "required": ["command"]}}}

    async def execute(self, args: dict) -> str:
        cmd = args.get("command", "")
        if not cmd: return "[Error: command required]"
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        if err: out += f"\n[stderr]\n{err}"
        if proc.returncode: out += f"\n[exit code: {proc.returncode}]"
        return out


class ReadHandler(McpHandler):
    @property
    def name(self) -> str: return "read"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "read", "description": "Read file contents with line numbers.", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}, "__tool_use_purpose": {"type": "string"}}, "required": ["filePath"]}}}

    async def execute(self, args: dict) -> str:
        fp = args.get("filePath") or args.get("path", "")
        if not fp: return "[Error: filePath required]"
        p = Path(fp).expanduser()
        if not p.exists(): return f"[Error: file not found: {p}]"
        if p.is_dir(): return f"[Error: {p} is a directory]"
        try:
            content = p.read_text(errors="replace")
        except Exception as e:
            return f"[Error reading {p}: {e}]"
        lines = content.splitlines(keepends=True)
        start = max(0, int(args.get("offset", 1)) - 1)
        limit = args.get("limit")
        if limit: lines = lines[start:start + int(limit)]
        else: lines = lines[start:]
        return "\n".join(f"{i+start+1:>6}\t{l.rstrip()}" for i, l in enumerate(lines)) or "[empty file]"


class WriteHandler(McpHandler):
    @property
    def name(self) -> str: return "write"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "write", "description": "Write content to a file, creating directories if needed.", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}, "__tool_use_purpose": {"type": "string"}}, "required": ["filePath", "content"]}}}

    async def execute(self, args: dict) -> str:
        fp = args.get("filePath") or args.get("path", "")
        content = args.get("content", "")
        if not fp: return "[Error: filePath required]"
        p = Path(fp).expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"[Wrote {len(content)} bytes to {p}]"
        except Exception as e:
            return f"[Error writing {p}: {e}]"


class EditHandler(McpHandler):
    @property
    def name(self) -> str: return "edit"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "edit", "description": "Replace text in a file.", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "path": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}, "replaceAll": {"type": "boolean"}, "__tool_use_purpose": {"type": "string"}}, "required": ["filePath", "oldString", "newString"]}}}

    async def execute(self, args: dict) -> str:
        fp = args.get("filePath") or args.get("path", "")
        old = args.get("oldString") or args.get("old_string", "")
        new = args.get("newString") or args.get("new_string", "")
        if not fp: return "[Error: filePath required]"
        if not old: return "[Error: oldString required]"
        p = Path(fp).expanduser()
        if not p.exists(): return f"[Error: file not found: {p}]"
        content = p.read_text(errors="replace")
        count = content.count(old)
        if count == 0: return f"[Error: oldString not found in {p}]"
        if args.get("replaceAll") or args.get("replace_all"):
            p.write_text(content.replace(old, new))
            return f"[Replaced {count} occurrence(s) in {p}]"
        if count > 1: return f"[Error: oldString found {count} times — use replaceAll or more context]"
        p.write_text(content.replace(old, new, 1))
        return f"[Replaced 1 occurrence in {p}]"


class GrepHandler(McpHandler):
    @property
    def name(self) -> str: return "grep"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "grep", "description": "Search for regex pattern in files.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "directory": {"type": "string"}, "include": {"type": "string"}, "exclude": {"type": "string"}, "__tool_use_purpose": {"type": "string"}}, "required": ["pattern"]}}}

    async def execute(self, args: dict) -> str:
        pattern = args.get("pattern", "")
        directory = args.get("path") or args.get("directory") or "."
        if not pattern: return "[Error: pattern required]"
        cmd = ["grep", "-rn", "--max-count=50"]
        if args.get("include"): cmd.extend(["--include", args["include"]])
        if args.get("exclude"): cmd.extend(["--exclude", args["exclude"]])
        cmd.extend([pattern, directory])
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            out = stdout.decode(errors="replace").strip()
            if proc.returncode == 1: return f"[No matches for '{pattern}']"
            if len(out) > 30000: out = "\n".join(out.splitlines()[:500]) + "\n...(truncated)"
            return out or f"[No matches]"
        except asyncio.TimeoutError: return "[grep timed out]"


class GlobHandler(McpHandler):
    @property
    def name(self) -> str: return "glob"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "glob", "description": "Find files by glob pattern.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "__tool_use_purpose": {"type": "string"}}, "required": ["pattern"]}}}

    async def execute(self, args: dict) -> str:
        import glob as _glob_mod
        pattern = args.get("pattern", "")
        base = Path(args.get("path") or ".").expanduser()
        if not pattern: return "[Error: pattern required]"
        if not base.exists(): return f"[Error: directory not found: {base}]"
        matches = sorted(_glob_mod.glob(str(base / pattern), recursive=True))
        if not matches: return f"[No files matching '{pattern}']"
        return "\n".join(matches[:200])


class FindHandler(McpHandler):
    @property
    def name(self) -> str: return "find"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "find", "description": "Find files by name or type.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "path": {"type": "string"}, "type": {"type": "string"}, "__tool_use_purpose": {"type": "string"}}, "required": []}}}

    async def execute(self, args: dict) -> str:
        cmd = ["find", args.get("path") or ".", "-maxdepth", "5"]
        if args.get("name"): cmd.extend(["-name", args["name"]])
        if args.get("type") in ("f", "d"): cmd.extend(["-type", args["type"]])
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            out = stdout.decode(errors="replace").strip()
            if not out: return "[No results]"
            lines = out.splitlines()
            if len(lines) > 200: return "\n".join(lines[:200]) + f"\n... ({len(lines)} total)"
            return out
        except asyncio.TimeoutError: return "[find timed out]"


class ListDirHandler(McpHandler):
    @property
    def name(self) -> str: return "list_dir"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "list_dir", "description": "List directory contents with sizes.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "__tool_use_purpose": {"type": "string"}}, "required": []}}}

    async def execute(self, args: dict) -> str:
        p = Path(args.get("path") or ".").expanduser()
        if not p.exists(): return f"[Error: path not found: {p}]"
        if not p.is_dir(): return f"[Error: {p} is not a directory]"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        lines = []
        for e in entries:
            if e.is_dir(): lines.append(f"DIR  {e.name}/")
            else:
                sz = e.stat().st_size
                sz_s = f"{sz}B" if sz < 1024 else f"{sz/1024:.1f}KB" if sz < 1048576 else f"{sz/1048576:.1f}MB"
                lines.append(f"FILE {e.name}  ({sz_s})")
        return "\n".join(lines) or "[empty directory]"


class WebFetchHandler(McpHandler):
    @property
    def name(self) -> str: return "web_fetch"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "web_fetch", "description": "Fetch content from a URL via HTTP GET.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "maxLength": {"type": "integer"}, "__tool_use_purpose": {"type": "string"}}, "required": ["url"]}}}

    async def execute(self, args: dict) -> str:
        url = args.get("url", "")
        if not url: return "[Error: url required]"
        max_len = args.get("maxLength") or args.get("max_length") or 50000
        try:
            proc = await asyncio.create_subprocess_exec("curl", "-sL", "-A", "Mozilla/5.0 (KiroCrew/1.0)", "--max-time", "30", "-o", "-", url, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=35.0)
            if proc.returncode != 0: return f"[Error fetching {url}]"
            content = stdout.decode(errors="replace")
            if len(content) > max_len: content = content[:max_len] + f"\n...(truncated at {max_len})"
            return content
        except asyncio.TimeoutError: return f"[Fetch timed out: {url}]"


class WebSearchHandler(McpHandler):
    @property
    def name(self) -> str: return "web_search"

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information using DuckDuckGo. Returns titles, URLs, and snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string"},
                        "max_results": {"type": "integer", "description": "Max results (1-10, default 5)", "default": 5},
                        "__tool_use_purpose": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }

    async def execute(self, args: dict) -> str:
        query = args.get("query", "")
        max_results = min(args.get("max_results", 5), 10)
        if not query:
            return "[Error: No query provided]"
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._search, query, max_results)
            if not results:
                return f"[No results found for: '{query}']"
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. **{r['title']}**\n   {r['href']}\n   {r['body']}")
            return "\n\n".join(lines)
        except Exception as e:
            return f"[Search error: {e}]"

    @staticmethod
    def _search(query: str, max_results: int) -> list:
        try:
            from ddgs import DDGS
            return DDGS().text(query, max_results=max_results)
        except ImportError:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))


# ── Gateway-proxied Handlers (ask_question, spawn_run, spawn_list) ───────────
# These are defined in kirocrew-core (mcp_core.py) but the OpenAI provider
# bypasses kiro-cli entirely, so we reimplement the key tools locally.
# ask_question builds the session_directive JSON directly (no gateway needed).
# spawn_run/spawn_list proxy to /api/spawn on the gateway HTTP API.


def _resolve_gateway_api() -> str:
    """Resolve the KiroCrew gateway API base URL from config (not hardcoded port)."""
    # Priority 1: explicit env var override
    port = os.environ.get("KIROCREW_PORT", "")
    if port:
        return f"http://localhost:{port}"
    # Priority 2: resolve from dashboard.url in kirocrew config
    try:
        from kiro_crew.config.loader import KiroCrewConfig
        from kiro_crew.dashboard.origin import parse_dashboard_url
        cfg = KiroCrewConfig.load()
        _host, port = parse_dashboard_url(cfg.dashboard.url)
        return f"http://localhost:{port}"
    except Exception:
        pass
    # Priority 3: scan for the gateway PID file / known port
    try:
        from kiro_crew.config.loader import config_dir
        for pf in sorted(config_dir().glob("gateway_*.port"), reverse=True):
            return f"http://localhost:{pf.read_text().strip()}"
    except Exception:
        pass
    # Fallback: best-effort
    return "http://localhost:5476"


def _resolve_session_key() -> str:
    """Resolve session key from env or PID file."""
    sk = os.environ.get("KIROCREW_SESSION_KEY", "")
    if sk:
        return sk
    try:
        from kiro_crew.config.loader import config_dir
        cfg_dir = config_dir()
        pid = os.getppid()
        seen = set()
        while pid > 1 and pid not in seen:
            seen.add(pid)
            pid_file = cfg_dir / f"session_pid_{pid}.txt"
            if pid_file.exists():
                key = pid_file.read_text().strip()
                if key:
                    return key
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            pid = int(line.split()[1])
                            break
            except Exception:
                break
    except Exception:
        pass
    return ""


def _internal_secret() -> str:
    """Read internal secret for gateway auth."""
    try:
        from kiro_crew.config.loader import config_dir
        return (config_dir() / ".local_secret").read_text().strip()
    except Exception:
        return ""


async def _gateway_post(path: str, body: dict) -> dict:
    """POST to KiroCrew gateway loopback API."""
    import urllib.request
    import urllib.error

    api = _resolve_gateway_api()
    data = json.dumps(body).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Secret": _internal_secret(),
    }
    sk = _resolve_session_key()
    if sk:
        headers["X-Session-Key"] = sk

    req = urllib.request.Request(f"{api}{path}", data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read()).get("error", str(e))
        except Exception:
            err = str(e)
        return {"error": err}
    except Exception as e:
        return {"error": str(e)}


# ── ask_question ─────────────────────────────────────────────────────────────

class AskQuestionHandler(McpHandler):
    """Returns a session_directive JSON that the chat_runner processes to
    render a question card. No gateway HTTP needed — pure local."""

    @property
    def name(self) -> str: return "ask_question"

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "ask_question",
                "description": (
                    "Ask the dashboard user 1-4 multiple-choice questions and pause "
                    "until they answer. Dashboard sessions only."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string", "description": "Question text (max 500)"},
                                    "header": {"type": "string", "description": "Category badge (max 50)"},
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "label": {"type": "string", "description": "Option text (max 200)"},
                                                "description": {"type": "string", "description": "Gloss (max 500)"},
                                            },
                                            "required": ["label"],
                                        },
                                    },
                                    "multiSelect": {"type": "boolean"},
                                },
                                "required": ["question", "options"],
                            },
                        },
                        "timeout_secs": {"type": "integer", "description": "Wait timeout (15-540, default 300)"},
                    },
                    "required": ["questions"],
                },
            },
        }

    async def execute(self, args: dict) -> str:
        questions = args.get("questions", [])
        if not questions:
            return "[Error: questions required]"

        directive = {
            "__kirocrew_directive__": "ask_question",
            "questions": questions,
            "timeout_secs": args.get("timeout_secs", 300),
        }
        return json.dumps(directive, ensure_ascii=False)


# ── spawn_run ────────────────────────────────────────────────────────────────

class SpawnRunHandler(McpHandler):
    """Proxies spawn_run to KiroCrew gateway HTTP API (POST /api/spawn)."""

    @property
    def name(self) -> str: return "spawn_run"

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "spawn_run",
                "description": (
                    "Spawn subagent(s) to run tasks in the background. "
                    "Returns immediately — results arrive as [Subagent completion event] "
                    "messages in your conversation. For parallel work, use 'tasks' array."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Single task description"},
                        "tasks": {"type": "array", "items": {"type": "string"}, "description": "Multiple tasks in parallel"},
                        "agent": {"type": "string", "description": "Agent name for subagent"},
                        "agents": {"type": "array", "items": {"type": "string"}, "description": "Agent names per task"},
                        "max_turns": {"type": "integer", "description": "Tool-call budget override"},
                        "cwd": {"type": "string", "description": "Working directory override"},
                        "model": {"type": "string", "description": "Model override"},
                        "keep": {"type": "boolean", "description": "Guarantee resumability"},
                        "include_memory": {"type": "boolean", "description": "Include memory context (default true)"},
                        "include_lessons": {"type": "boolean", "description": "Include lessons (default true)"},
                        "include_project": {"type": "boolean", "description": "Include project context (default true)"},
                    },
                },
            },
        }

    async def execute(self, args: dict) -> str:
        tasks = args.get("tasks")
        task = args.get("task")

        if tasks and isinstance(tasks, list):
            task_list = [t for t in tasks if isinstance(t, str) and t.strip()]
        elif task:
            task_list = [task]
        else:
            return "Error: task or tasks is required"

        parent_session = _resolve_session_key()
        agent = args.get("agent") or ""
        agents_list = args.get("agents") or []
        max_turns = args.get("max_turns") or 0
        cwd = args.get("cwd") or ""
        model = args.get("model") or ""
        keep = bool(args.get("keep"))
        inc_memory = args.get("include_memory", True) is not False
        inc_lessons = args.get("include_lessons", True) is not False
        inc_project = args.get("include_project", True) is not False

        if agents_list and len(agents_list) != len(task_list):
            return f"Error: agents length ({len(agents_list)}) must match tasks ({len(task_list)})"

        agent_ids = []
        agent_names = []
        errors = []

        for i, t in enumerate(task_list):
            a = agents_list[i] if agents_list else agent
            body: dict[str, Any] = {"task": t, "agent": a, "parent_session": parent_session}
            if max_turns: body["max_turns"] = max_turns
            if cwd: body["cwd"] = cwd
            if model: body["model"] = model
            if keep: body["keep"] = True
            if not inc_memory: body["include_memory"] = False
            if not inc_lessons: body["include_lessons"] = False
            if not inc_project: body["include_project"] = False

            d = await _gateway_post("/api/spawn", body)
            if d.get("error"):
                errors.append(f"{t[:60]}: {d['error']}")
            else:
                agent_ids.append(d.get("id", "?"))
                agent_names.append(a)

        lines = []
        if agent_ids:
            lines.append(f"Spawned {len(agent_ids)} subagent(s). Results will arrive as completion events:")
            for aid, a, t in zip(agent_ids, agent_names, task_list):
                label = f"{aid} ({a})" if a else aid
                lines.append(f"  {label}: {t[:80]}")
            lines.append("\n⚠️ END YOUR TURN NOW — wait for [Subagent completion event] messages.")
        if errors:
            lines.append(f"\n❌ {len(errors)} task(s) failed:")
            for e in errors:
                lines.append(f"  - {e}")
        if not agent_ids and not errors:
            lines.append("Error: no subagents started.")

        return "\n".join(lines)


# ── spawn_list ───────────────────────────────────────────────────────────────

class SpawnListHandler(McpHandler):
    @property
    def name(self) -> str: return "spawn_list"

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": "spawn_list", "description": "List running/completed subagents.", "parameters": {"type": "object", "properties": {}}}}

    async def execute(self, args: dict) -> str:
        d = await _gateway_post("/api/spawn", {})
        agents = d.get("agents", [])
        if not agents:
            return "No subagents running."
        lines = []
        for a in agents:
            status = "done" if a.get("done") else "running"
            err = f" error: {a['error']}" if a.get("error") else ""
            lines.append(f"{a['id']}  [{status}]{err}  {a.get('task', '')[:60]}")
        return "\n".join(lines)


# ── Auto-register all handlers ───────────────────────────────────────────────

def register_all_builtins() -> None:
    """Register all local handlers. Call once at startup."""
    McpHandlerRegistry.register_many(
        BashHandler(),
        ReadHandler(),
        WriteHandler(),
        EditHandler(),
        GrepHandler(),
        GlobHandler(),
        FindHandler(),
        ListDirHandler(),
        WebFetchHandler(),
        WebSearchHandler(),
        # Gateway-proxied handlers
        AskQuestionHandler(),
        SpawnRunHandler(),
        SpawnListHandler(),
    )
    logger.info("McpHandlerRegistry: registered %d handlers", len(McpHandlerRegistry._handlers))
