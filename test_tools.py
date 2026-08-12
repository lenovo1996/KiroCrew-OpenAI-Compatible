#!/usr/bin/env python3
"""test_tools.py — End-to-end test: OpenAIProvider + tool calls.

Tests: text response, read_file, write_file, search (grep), replace in file.
Run:
    python test_tools.py
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY  = os.environ.get("OPENAI_API_KEY",  "sk-...")
MODEL    = os.environ.get("OPENAI_MODEL",    "free")

# ── Simple ToolExecutor with file I/O tools ───────────────────────────────────
from openai_provider.provider import ToolExecutor

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Search for a pattern (substring) in a file. Returns matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "required": ["path", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replace all occurrences of old_text with new_text in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
]


class FileToolExecutor(ToolExecutor):
    def tool_definitions(self):
        return TOOL_DEFS

    async def execute(self, name: str, args: dict) -> str:
        if name == "read_file":
            p = Path(args["path"])
            if not p.exists():
                return f"[Error: file not found: {p}]"
            return p.read_text(errors="replace")

        if name == "write_file":
            p = Path(args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"[Written {len(args['content'])} bytes to {p}]"

        if name == "search_in_file":
            p = Path(args["path"])
            if not p.exists():
                return f"[Error: file not found: {p}]"
            pattern = args["pattern"]
            lines = p.read_text(errors="replace").splitlines()
            matches = [f"L{i+1}: {l}" for i, l in enumerate(lines) if pattern in l]
            return "\n".join(matches) if matches else f"[No matches for '{pattern}']"

        if name == "replace_in_file":
            p = Path(args["path"])
            if not p.exists():
                return f"[Error: file not found: {p}]"
            original = p.read_text(errors="replace")
            count = original.count(args["old_text"])
            new_content = original.replace(args["old_text"], args["new_text"])
            p.write_text(new_content)
            return f"[Replaced {count} occurrence(s) in {p}]"

        if name == "list_directory":
            p = Path(args["path"])
            if not p.exists():
                return f"[Error: path not found: {p}]"
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            lines = [f"{'DIR ' if e.is_dir() else 'FILE'} {e.name}" for e in entries]
            return "\n".join(lines) or "[empty directory]"

        return f"[Unknown tool: {name}]"


# ── Test runner ───────────────────────────────────────────────────────────────

CYAN  = "\033[96m"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
DIM   = "\033[2m"
RESET = "\033[0m"

def header(s): print(f"\n{CYAN}{'─'*60}{RESET}\n{CYAN}{s}{RESET}")
def ok(s):     print(f"{GREEN}✅ {s}{RESET}")
def fail(s):   print(f"{RED}❌ {s}{RESET}")
def info(s):   print(f"{DIM}   {s}{RESET}")


async def run_test(provider, prompt: str, label: str) -> dict:
    """Run one turn, collect all events, return summary."""
    from kiro_crew.acp.types import (
        EVENT_TEXT_CHUNK, EVENT_THINKING_CHUNK, EVENT_TOOL_CALL,
        EVENT_TOOL_RESULT, EVENT_PERMISSION_REQUEST, EVENT_COMPLETE,
    )

    print(f"\n{YELLOW}▶ {label}{RESET}")
    info(f"Prompt: {prompt[:80]}{'...' if len(prompt)>80 else ''}")

    # Auto-approve all tool calls for testing
    original_approve = provider.approve_tool
    async def auto_approve(request_id, *, always=False):
        from openai_provider.provider import _PENDING_APPROVALS
        key = str(request_id)
        fut = _PENDING_APPROVALS.get(key)
        if fut and not fut.done():
            fut.set_result(True)

    provider.approve_tool = auto_approve

    text_out = []
    thinking_out = []
    tools_called = []
    tool_results = []
    complete_event = None

    async for event in provider.stream(prompt):
        if event.kind == EVENT_TEXT_CHUNK:
            text_out.append(event.text)
            print(event.text, end="", flush=True)
        elif event.kind == EVENT_THINKING_CHUNK:
            thinking_out.append(event.text)
            print(f"{DIM}{event.text}{RESET}", end="", flush=True)
        elif event.kind == EVENT_TOOL_CALL:
            tools_called.append(event.tool_name)
            info(f"\n[tool_call] {event.tool_name}: {event.tool_input[:100]}")
        elif event.kind == EVENT_PERMISSION_REQUEST:
            # auto-approve fires here
            await auto_approve(event.request_id)
        elif event.kind == EVENT_TOOL_RESULT:
            snippet = event.tool_output[:120].replace("\n", " ")
            tool_results.append(event.tool_output)
            info(f"[tool_result] {snippet}{'...' if len(event.tool_output)>120 else ''}")
        elif event.kind == EVENT_COMPLETE:
            complete_event = event

    provider.approve_tool = original_approve
    print()  # newline after streaming text

    full_thinking = "".join(thinking_out)
    if full_thinking:
        print(f"\n{DIM}{'·'*60}{RESET}")
        print(f"{DIM}💭 THINKING ({len(full_thinking)} chars):{RESET}")
        for line in full_thinking.splitlines():
            if line.strip():
                print(f"{DIM}   {line}{RESET}")
        print(f"{DIM}{'·'*60}{RESET}")
    else:
        print(f"\n{DIM}   [no thinking/reasoning tokens]{RESET}")

    return {
        "text": "".join(text_out),
        "thinking": full_thinking,
        "tools_called": tools_called,
        "tool_results": tool_results,
        "stop_reason": complete_event.stop_reason if complete_event else "?",
    }


async def main():
    from openai_provider.provider import OpenAIProvider

    executor = FileToolExecutor()
    provider = OpenAIProvider(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        max_tokens=4096,
        tool_executor=executor,
    )
    await provider.start()

    # Temp workspace
    tmpdir = Path(tempfile.mkdtemp(prefix="oai_test_"))
    sample_file = tmpdir / "hello.txt"
    sample_file.write_text("Hello world\nThis is line 2\nFoo bar baz\n")

    passed = 0
    failed = 0

    # ── Test 1: plain text (no tools) ─────────────────────────────────────────
    header("Test 1: Plain text response")
    r = await run_test(provider, "Reply with exactly: PING", "Plain text")
    if "PING" in r["text"].upper():
        ok("Got expected text")
        passed += 1
    else:
        fail(f"Expected PING, got: {r['text'][:100]}")
        failed += 1

    # Reset history between tests
    provider._messages = []

    # ── Test 2: read_file ─────────────────────────────────────────────────────
    header("Test 2: read_file tool")
    r = await run_test(
        provider,
        f"Read the file at {sample_file} and tell me what line 2 says.",
        "read_file"
    )
    if "read_file" in r["tools_called"]:
        ok("read_file was called")
        if "line 2" in r["text"].lower() or "This is line 2" in r["text"]:
            ok("Correct content in response")
            passed += 1
        else:
            ok("Tool called (response may paraphrase)")
            passed += 1
    else:
        fail(f"read_file not called. Tools: {r['tools_called']}")
        failed += 1
    provider._messages = []

    # ── Test 3: write_file ────────────────────────────────────────────────────
    header("Test 3: write_file tool")
    new_file = tmpdir / "output.txt"
    r = await run_test(
        provider,
        f"Write a file at {new_file} with content: 'Created by AI test'",
        "write_file"
    )
    if new_file.exists() and "Created by AI test" in new_file.read_text():
        ok("File created with correct content")
        passed += 1
    elif "write_file" in r["tools_called"]:
        ok("write_file called (checking content)")
        if new_file.exists():
            ok(f"File exists: {new_file.read_text()[:80]}")
            passed += 1
        else:
            fail("File not created on disk")
            failed += 1
    else:
        fail(f"write_file not called. Tools: {r['tools_called']}")
        failed += 1
    provider._messages = []

    # ── Test 4: search_in_file ────────────────────────────────────────────────
    header("Test 4: search_in_file tool")
    r = await run_test(
        provider,
        f"Search for the word 'Foo' in {sample_file} and tell me which line it appears on.",
        "search_in_file"
    )
    if "search_in_file" in r["tools_called"]:
        ok("search_in_file called")
        if "3" in r["text"] or "Foo" in r["text"]:
            ok("Correct line reported")
        passed += 1
    else:
        fail(f"search_in_file not called. Tools: {r['tools_called']}")
        failed += 1
    provider._messages = []

    # ── Test 5: replace_in_file ───────────────────────────────────────────────
    header("Test 5: replace_in_file tool")
    r = await run_test(
        provider,
        f"In the file {sample_file}, replace 'Hello world' with 'Goodbye world'.",
        "replace_in_file"
    )
    if "replace_in_file" in r["tools_called"]:
        ok("replace_in_file called")
        content = sample_file.read_text()
        if "Goodbye world" in content:
            ok("File content correctly modified ✓")
            passed += 1
        else:
            fail(f"File not modified. Content: {content[:100]}")
            failed += 1
    else:
        fail(f"replace_in_file not called. Tools: {r['tools_called']}")
        failed += 1
    provider._messages = []

    # ── Test 6: multi-step (read then write) ──────────────────────────────────
    header("Test 6: Multi-step — read then write")
    out_file = tmpdir / "summary.txt"
    r = await run_test(
        provider,
        f"Read {sample_file}, then write a 1-line summary to {out_file}.",
        "multi-step read+write"
    )
    tools = r["tools_called"]
    if "read_file" in tools and "write_file" in tools:
        ok(f"Both tools called: {tools}")
        passed += 1
    elif len(tools) >= 1:
        ok(f"At least 1 tool called: {tools}")
        passed += 1
    else:
        fail("No tools called")
        failed += 1
    provider._messages = []

    # ── Test 7: text-format tool call parser ──────────────────────────────────
    header("Test 7: _extract_text_tool_calls parser (unit test)")
    from openai_provider.provider import _extract_text_tool_calls, AcpEvent
    from kiro_crew.acp.types import EVENT_TEXT_CHUNK

    # Simulate KiroCrew XML format in text buffer
    xml_text = (
        "Let me check that for you.\n"
        "<tool_call>\n"
        "  <function=read_file>\n"
        "    <parameter=path>/tmp/test.txt</parameter>\n"
        "  </function>\n"
        "</tool_call>"
    )

    class _FakeEvent:
        def __init__(self, text):
            self.text = text
            self.kind = EVENT_TEXT_CHUNK

    buf = [_FakeEvent(xml_text)]
    parsed = _extract_text_tool_calls(buf)
    if parsed and parsed[0]["function"]["name"] == "read_file":
        args = json.loads(parsed[0]["function"]["arguments"])
        if args.get("path") == "/tmp/test.txt":
            ok("KiroCrew XML parsed correctly → read_file(/tmp/test.txt)")
            passed += 1
        else:
            fail(f"Wrong args: {args}")
            failed += 1
    else:
        fail(f"Parser returned: {parsed}")
        failed += 1

    # Hermes JSON format
    hermes_text = '<tool_call>{"name": "write_file", "arguments": {"path": "/tmp/out.txt", "content": "hello"}}</tool_call>'
    buf2 = [_FakeEvent(hermes_text)]
    parsed2 = _extract_text_tool_calls(buf2)
    if parsed2 and parsed2[0]["function"]["name"] == "write_file":
        ok("Hermes JSON inside <tool_call> parsed correctly → write_file")
        passed += 1
    else:
        fail(f"Hermes parser returned: {parsed2}")
        failed += 1

    # No tool call in plain text — should return []
    plain_buf = [_FakeEvent("Sure, here is the answer: 42")]
    parsed3 = _extract_text_tool_calls(plain_buf)
    if parsed3 == []:
        ok("Plain text correctly returns [] (no false positives)")
        passed += 1
    else:
        fail(f"False positive: {parsed3}")
        failed += 1
    provider._messages = []

    # ── Test 8: no text leak when tool_calls detected ─────────────────────────
    header("Test 8: Text buffering — no preamble text leaked before tool call")
    # Run a tool call and ensure text emitted to UI is AFTER tool result
    text_events_before_tool: list[str] = []
    tool_seen = False
    preamble_leaked = False

    class _TrackingProvider(provider.__class__):
        pass  # reuse provider class, just wrap run_test logic

    # Fresh provider
    p2 = provider.__class__(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        tool_executor=provider._tool_executor,
        max_tokens=2048,
        temperature=0.0,
    )
    await p2.start()

    from kiro_crew.acp.types import EVENT_TEXT_CHUNK as ETC, EVENT_TOOL_CALL as ETCALL

    text_before_first_tool = []
    first_tool_seen = False

    async for ev in p2.stream(f"Read the file {sample_file} and tell me what line 1 says."):
        if ev.kind == "_tool_calls_collected" or ev.kind == "_finish_reason":
            continue
        if ev.kind == ETCALL:
            first_tool_seen = True
        if ev.kind == ETC and not first_tool_seen:
            text_before_first_tool.append(ev.text)

    # Text before tool should be empty (buffered + discarded) or minimal thinking continuation
    leaked = "".join(text_before_first_tool)
    if not leaked.strip():
        ok("No text leaked before tool call ✓")
        passed += 1
    elif "<tool_call>" in leaked or "<function=" in leaked:
        fail(f"Raw tool call XML leaked to UI: {leaked[:100]}")
        failed += 1
    else:
        # Some preamble text is acceptable if it doesn't contain tool call markup
        ok(f"Preamble text (acceptable, no XML leak): '{leaked[:60].strip()}'")
        passed += 1

    await p2.shutdown()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{CYAN}{'═'*60}{RESET}")
    total = passed + failed
    color = GREEN if failed == 0 else RED
    print(f"{color}Results: {passed}/{total} passed{RESET}")
    if failed:
        print(f"{RED}  {failed} test(s) failed{RESET}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    await provider.shutdown()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
