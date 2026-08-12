"""mcp_executor.py — ToolExecutor that routes tool calls through KiroCrew's MCP layer.

KiroCrew manages its own MCP servers (kirocrew-core, kirocrew-cron, file tools,
bash, etc.). This executor exposes those tools to the OpenAI model as standard
function-calling definitions, then routes execution back through the MCP client.

The MCP tool schema is fetched lazily on first use so the executor doesn't need
a live session at import time.

Built-in fallback tools (read, write, edit, grep, glob, find, list_dir,
web_fetch, web_search) are handled locally when no MCP server is available.
This is necessary because kiro-cli's built-in tools (written in Go) are not
MCP servers — they only exist inside the ACP runtime. The OpenAI provider
bypasses kiro-cli entirely, so these tools must be reimplemented as Python
handlers.
"""

from __future__ import annotations

import asyncio
import glob as _glob_mod
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .provider import ToolExecutor

logger = logging.getLogger(__name__)


def _mcp_tool_to_openai(tool: dict) -> dict:
    """Convert an MCP tool definition to OpenAI function-calling format."""
    input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    # Strip additionalProperties if present (some MCP servers add it, OpenAI rejects)
    clean_schema = {k: v for k, v in input_schema.items() if k != "additionalProperties"}
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": clean_schema or {"type": "object", "properties": {}},
        },
    }


class McpToolExecutor(ToolExecutor):
    """Executor that exposes KiroCrew MCP tools to the OpenAI model.

    Reads available tools from the MCP gateway's shared server pool (if enabled)
    or from the per-session MCP clients. Falls back gracefully if MCP is
    unavailable.
    """

    def __init__(self) -> None:
        self._tools_cache: list[dict] | None = None
        self._definitions_cache: list[dict] | None = None
        self._lock = asyncio.Lock()

    # ── Tool discovery ────────────────────────────────────────────────────────

    def tool_definitions(self) -> list[dict]:
        """Return OpenAI tool definitions (sync — uses cached value)."""
        if self._definitions_cache is not None:
            return self._definitions_cache
        # Sync fetch on first call (blocking but one-time)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't block — return empty and populate on next async call
                asyncio.ensure_future(self._fetch_tools_async())
                return []
            else:
                loop.run_until_complete(self._fetch_tools_async())
                return self._definitions_cache or []
        except Exception:
            return []

    async def _fetch_tools_async(self) -> None:
        """Populate tools cache by querying installed MCP servers."""
        async with self._lock:
            if self._definitions_cache is not None:
                return
            tools: list[dict] = []
            try:
                tools = await _list_mcp_tools()
            except Exception as exc:
                logger.warning("McpToolExecutor: could not list MCP tools: %s", exc)
                tools = []

            # Always merge in built-in tool definitions (MCP servers may not
            # expose file/web tools that kiro-cli normally provides)
            builtin = _builtin_tool_definitions()
            builtin_names = {t["name"] for t in builtin}
            # MCP tools take priority; only add built-ins not already present
            mcp_names = {t["name"] for t in tools}
            for bt in builtin:
                if bt["name"] not in mcp_names:
                    tools.append(bt)

            self._tools_cache = tools
            self._definitions_cache = [_mcp_tool_to_openai(t) for t in tools]
            logger.info(
                "McpToolExecutor: loaded %d tool definitions (%d MCP + %d built-in)",
                len(self._definitions_cache),
                len(tools) - len(builtin) + len(builtin_names & mcp_names),
                len(builtin_names - mcp_names),
            )

    # ── Tool execution ────────────────────────────────────────────────────────

    async def execute(self, name: str, args: dict) -> Any:
        """Execute a tool by name with given args, return string result."""
        # Ensure tools are loaded
        if self._definitions_cache is None:
            await self._fetch_tools_async()

        try:
            result = await _call_mcp_tool(name, args)
            return result
        except Exception as exc:
            logger.exception("McpToolExecutor: tool %s failed", name)
            return f"[Tool execution error: {exc}]"


# ── MCP client helpers ────────────────────────────────────────────────────────

async def _list_mcp_tools() -> list[dict]:
    """List tools from the MCP gateway or installed servers."""
    # Try the shared gateway first
    try:
        from kiro_crew.mcp_gateway.session_servers import get_pooled_servers
        servers = get_pooled_servers()
        tools = []
        for server in servers.values():
            try:
                resp = await server.list_tools()
                tools.extend(resp.get("tools", []))
            except Exception:
                pass
        if tools:
            return tools
    except ImportError:
        pass

    # Fallback: spawn a minimal MCP client and list managed servers
    try:
        from kiro_crew.mcp_providers.official import OfficialMcpProvider
        provider = OfficialMcpProvider()
        tools = await provider.list_tools()
        return tools
    except Exception:
        pass

    return []


async def _call_mcp_tool(name: str, args: dict) -> str:
    """Call a tool via MCP and return the text result.

    Routing order:
      1. MCP gateway (pooled servers)
      2. Built-in tool handlers (file I/O, web, shell)
    """
    # Try gateway first
    try:
        from kiro_crew.mcp_gateway.session_servers import get_pooled_servers
        servers = get_pooled_servers()
        for server in servers.values():
            try:
                resp = await server.call_tool(name, args)
                content = resp.get("content", [])
                parts = []
                for block in content:
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "\n".join(parts) if parts else json.dumps(resp)
            except Exception:
                continue
    except ImportError:
        pass

    # ── Built-in tool handlers ────────────────────────────────────────────────
    # These reimplement kiro-cli's built-in tools in Python for the OpenAI
    # provider path (kiro-cli is bypassed when using gateway.py).

    if name in ("execute_bash", "shell", "bash"):
        cmd = args.get("command", "")
        if cmd:
            return await _run_bash(cmd)

    if name == "read":
        return await _tool_read(args)

    if name in ("write", "create_file"):
        return await _tool_write(args)

    if name == "edit":
        return await _tool_edit(args)

    if name == "grep":
        return await _tool_grep(args)

    if name == "glob":
        return await _tool_glob(args)

    if name == "find":
        return await _tool_find(args)

    if name == "list_dir":
        return await _tool_list_dir(args)

    if name == "web_fetch":
        return await _tool_web_fetch(args)

    if name == "web_search":
        return await _tool_web_search(args)

    raise RuntimeError(f"No MCP server found for tool: {name}")


# ── Built-in tool implementations ─────────────────────────────────────────────
# These mirror kiro-cli's built-in tools for the OpenAI provider path.

async def _tool_read(args: dict) -> str:
    """Read file contents. Mirrors kiro-cli's 'read' tool.

    Args:
        filePath or path: file path to read
        offset: optional start line (1-based)
        limit: optional max lines to read
    """
    file_path = args.get("filePath") or args.get("path", "")
    if not file_path:
        return "[Error: filePath or path is required]"

    p = Path(file_path).expanduser()
    if not p.exists():
        return f"[Error: file not found: {p}]"
    if p.is_dir():
        return f"[Error: {p} is a directory, not a file. Use list_dir instead.]"

    try:
        content = p.read_text(errors="replace")
    except Exception as exc:
        return f"[Error reading {p}: {exc}]"

    lines = content.splitlines(keepends=True)
    offset = args.get("offset")
    limit = args.get("limit")

    if offset is not None:
        try:
            offset = int(offset) - 1  # 1-based to 0-based
            if offset < 0:
                offset = 0
        except (ValueError, TypeError):
            offset = 0

    if limit is not None:
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = None

    start = offset if offset else 0
    if limit:
        lines = lines[start:start + limit]
    else:
        lines = lines[start:]

    # Add line numbers like kiro-cli does
    numbered = []
    for i, line in enumerate(lines, start=start + 1):
        numbered.append(f"{i:>6}\t{line.rstrip()}")

    result = "\n".join(numbered)
    if not result:
        return "[empty file]"
    return result


async def _tool_write(args: dict) -> str:
    """Write content to a file. Mirrors kiro-cli's 'write' tool.

    Args:
        filePath or path: target file path
        content: content to write
    """
    file_path = args.get("filePath") or args.get("path", "")
    content = args.get("content", "")

    if not file_path:
        return "[Error: filePath or path is required]"

    p = Path(file_path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"[Wrote {len(content)} bytes to {p}]"
    except Exception as exc:
        return f"[Error writing {p}: {exc}]"


async def _tool_edit(args: dict) -> str:
    """Replace text in a file. Mirrors kiro-cli's 'edit' tool.

    Args:
        filePath or path: target file path
        oldString: text to find (exact match)
        newString: replacement text
       replaceAll: optional, replace all occurrences (default: false)
    """
    file_path = args.get("filePath") or args.get("path", "")
    old_string = args.get("oldString") or args.get("old_string", "")
    new_string = args.get("newString") or args.get("new_string", "")
    replace_all = args.get("replaceAll") or args.get("replace_all", False)

    if not file_path:
        return "[Error: filePath or path is required]"
    if not old_string:
        return "[Error: oldString is required]"

    p = Path(file_path).expanduser()
    if not p.exists():
        return f"[Error: file not found: {p}]"

    try:
        content = p.read_text(errors="replace")
    except Exception as exc:
        return f"[Error reading {p}: {exc}]"

    count = content.count(old_string)
    if count == 0:
        return f"[Error: oldString not found in {p}]"

    if replace_all:
        new_content = content.replace(old_string, new_string)
        p.write_text(new_content)
        return f"[Replaced {count} occurrence(s) in {p}]"
    else:
        if count > 1:
            return (
                f"[Error: oldString found {count} times in {p}. "
                f"Provide more context to make it unique, or set replaceAll=true.]"
            )
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content)
        return f"[Replaced 1 occurrence in {p}]"


async def _tool_grep(args: dict) -> str:
    """Search for a pattern in files. Mirrors kiro-cli's 'grep' tool.

    Args:
        pattern: search pattern (regex)
        path or directory: directory to search in (default: cwd)
        include: optional glob filter (e.g. "*.py")
        exclude: optional glob exclusion
    """
    pattern = args.get("pattern", "")
    directory = args.get("path") or args.get("directory") or args.get("cwd", ".")
    include = args.get("include", "")
    exclude = args.get("exclude", "")

    if not pattern:
        return "[Error: pattern is required]"

    # Use ripgrep if available, fall back to grep
    rg_path = subprocess.run(["which", "rg"], capture_output=True, text=True).stdout.strip()
    if rg_path:
        cmd = ["rg", "--no-heading", "-n", "--max-count", "50"]
        if include:
            cmd.extend(["-g", include])
        if exclude:
            cmd.extend(["-g", f"!{exclude}"])
        cmd.extend([pattern, directory])
    else:
        cmd = ["grep", "-rn", "--max-count=50"]
        if include:
            cmd.extend(["--include", include])
        if exclude:
            cmd.extend(["--exclude", exclude])
        cmd.extend([pattern, directory])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        output = stdout.decode(errors="replace").strip()
        if proc.returncode == 1:
            return f"[No matches for '{pattern}']"
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            return f"[grep error: {err}]"
        # Limit output to prevent context flood
        if len(output) > 30000:
            lines = output.splitlines()
            output = "\n".join(lines[:500]) + f"\n... (truncated, {len(lines)} total matches)"
        return output or f"[No matches for '{pattern}']"
    except asyncio.TimeoutError:
        return "[Error: grep timed out (15s)]"
    except Exception as exc:
        return f"[Error running grep: {exc}]"


async def _tool_glob(args: dict) -> str:
    """Find files matching a glob pattern. Mirrors kiro-cli's 'glob' tool.

    Args:
        pattern: glob pattern (e.g. "**/*.py")
        path: optional base directory (default: cwd)
    """
    pattern = args.get("pattern", "")
    base = args.get("path") or args.get("directory") or "."

    if not pattern:
        return "[Error: pattern is required]"

    base_path = Path(base).expanduser()
    if not base_path.exists():
        return f"[Error: directory not found: {base_path}]"

    full_pattern = str(base_path / pattern)
    try:
        matches = sorted(_glob_mod.glob(full_pattern, recursive=True))
        if not matches:
            return f"[No files matching '{pattern}' in {base}]"
        # Limit results
        if len(matches) > 200:
            result = "\n".join(matches[:200]) + f"\n... ({len(matches)} total, showing first 200)"
        else:
            result = "\n".join(matches)
        return result
    except Exception as exc:
        return f"[Error running glob: {exc}]"


async def _tool_find(args: dict) -> str:
    """Find files by name or type. Mirrors kiro-cli's 'find' tool.

    Args:
        name: filename pattern (e.g. "*.py")
        path: directory to search in (default: cwd)
        type: "f" for files, "d" for directories
    """
    name_pattern = args.get("name", "")
    search_path = args.get("path") or args.get("directory") or "."
    file_type = args.get("type", "")

    cmd = ["find", search_path, "-maxdepth", "5"]
    if name_pattern:
        cmd.extend(["-name", name_pattern])
    if file_type in ("f", "d"):
        cmd.extend(["-type", file_type])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
        output = stdout.decode(errors="replace").strip()
        if not output:
            return "[No results found]"
        lines = output.splitlines()
        if len(lines) > 200:
            return "\n".join(lines[:200]) + f"\n... ({len(lines)} total, showing first 200)"
        return output
    except asyncio.TimeoutError:
        return "[Error: find timed out (15s)]"
    except Exception as exc:
        return f"[Error running find: {exc}]"


async def _tool_list_dir(args: dict) -> str:
    """List directory contents. Mirrors kiro-cli's 'list_dir' tool.

    Args:
        path: directory path to list (default: cwd)
    """
    dir_path = args.get("path") or args.get("directory") or "."
    p = Path(dir_path).expanduser()

    if not p.exists():
        return f"[Error: path not found: {p}]"
    if not p.is_dir():
        return f"[Error: {p} is not a directory]"

    try:
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        lines = []
        for entry in entries:
            if entry.is_dir():
                lines.append(f"DIR  {entry.name}/")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                lines.append(f"FILE {entry.name}  ({size_str})")
        return "\n".join(lines) or "[empty directory]"
    except Exception as exc:
        return f"[Error listing {p}: {exc}]"


async def _tool_web_fetch(args: dict) -> str:
    """Fetch content from a URL. Mirrors kiro-cli's 'web_fetch' tool.

    Args:
        url: URL to fetch
        maxLength: optional max response length (default: 50000)
    """
    url = args.get("url", "")
    if not url:
        return "[Error: url is required]"

    max_length = args.get("maxLength") or args.get("max_length") or 50000

    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-sL", "-A", "Mozilla/5.0 (KiroCrew/1.0)",
            "--max-time", "30", "-o", "-", url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=35.0)

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            return f"[Error fetching {url}: {err}]"

        content = stdout.decode(errors="replace")
        if len(content) > max_length:
            content = content[:max_length] + f"\n\n... (truncated at {max_length} chars)"
        return content
    except asyncio.TimeoutError:
        return f"[Error: fetch timed out for {url}]"
    except Exception as exc:
        return f"[Error fetching {url}: {exc}]"


async def _tool_web_search(args: dict) -> str:
    """Search the web. Returns a note that web search is not available.

    Args:
        query: search query
    """
    query = args.get("query", "")
    return (
        f"[Web search is not available in this environment. "
        f"Query was: '{query}'. Use web_fetch to fetch specific URLs instead.]"
    )


# ── Shell execution ───────────────────────────────────────────────────────────

async def _run_bash(command: str) -> str:
    """Execute a shell command (fallback for bash tools)."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace")
    result = out
    if err:
        result += f"\n[stderr]\n{err}"
    if proc.returncode and proc.returncode != 0:
        result += f"\n[exit code: {proc.returncode}]"
    return result


# ── Built-in tool definitions (fallback when MCP is unavailable) ──────────────

def _builtin_tool_definitions() -> list[dict]:
    """Built-in tool set — reimplements kiro-cli's built-in tools for the
    OpenAI provider path where kiro-cli is bypassed."""
    return [
        {
            "name": "execute_bash",
            "description": "Run a shell command and return its output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "read",
            "description": (
                "Read the contents of a file. Returns file content with line numbers. "
                "Use offset/limit for large files."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Absolute path to the file"},
                    "path": {"type": "string", "description": "Alternative: file path"},
                    "offset": {"type": "integer", "description": "Start line (1-based)"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": ["filePath"],
            },
        },
        {
            "name": "write",
            "description": "Write content to a file, creating directories if needed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Target file path"},
                    "path": {"type": "string", "description": "Alternative: file path"},
                    "content": {"type": "string", "description": "Content to write"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": ["filePath", "content"],
            },
        },
        {
            "name": "edit",
            "description": (
                "Replace text in a file. oldString must exactly match text in the file. "
                "Fails if multiple matches found unless replaceAll is true."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Target file path"},
                    "path": {"type": "string", "description": "Alternative: file path"},
                    "oldString": {"type": "string", "description": "Text to find"},
                    "newString": {"type": "string", "description": "Replacement text"},
                    "replaceAll": {"type": "boolean", "description": "Replace all occurrences"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": ["filePath", "oldString", "newString"],
            },
        },
        {
            "name": "grep",
            "description": "Search for a regex pattern in files. Returns matching lines with filenames and line numbers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in"},
                    "directory": {"type": "string", "description": "Alternative: directory path"},
                    "include": {"type": "string", "description": "Glob filter (e.g. *.py)"},
                    "exclude": {"type": "string", "description": "Glob exclusion"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "glob",
            "description": "Find files matching a glob pattern (e.g. '**/*.py').",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "description": "Base directory (default: cwd)"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "find",
            "description": "Find files by name or type in a directory tree.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Filename pattern (e.g. *.py)"},
                    "path": {"type": "string", "description": "Directory to search in"},
                    "type": {"type": "string", "description": "File type: f=file, d=directory"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": [],
            },
        },
        {
            "name": "list_dir",
            "description": "List files and directories at the given path with sizes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": [],
            },
        },
        {
            "name": "web_fetch",
            "description": "Fetch content from a URL via HTTP GET.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "maxLength": {"type": "integer", "description": "Max response chars (default: 50000)"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": ["url"],
            },
        },
        {
            "name": "web_search",
            "description": "Search the web for information.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "__tool_use_purpose": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    ]
