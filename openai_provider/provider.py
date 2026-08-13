"""OpenAIProvider — implements LLMProvider over any OpenAI-compatible API.

Maps OpenAI streaming chat-completion events → AcpEvent stream so KiroCrew's
TurnDriver / dashboard can consume it without modification.

Supported features:
  - text streaming          → EVENT_TEXT_CHUNK
  - tool_calls              → EVENT_TOOL_CALL + EVENT_PERMISSION_REQUEST
  - reasoning/thinking      → EVENT_THINKING_CHUNK  (o1/o3/deepseek-r1 style)
  - tool results            → EVENT_TOOL_RESULT
  - context usage           → context_usage_pct via prompt_tokens / context_window
  - cancel                  → asyncio.Event to abort in-flight stream

Config separation:
  - max_tokens      → output token limit per turn (sent as `max_tokens` in API call)
  - context_window  → total model context window (for usage reporting only)
                      auto-resolved from /v1/models at startup, overridable via
                      OPENAI_CONTEXT_WINDOW env var or constructor param.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re as _re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from kiro_crew.acp.types import (
    AcpEvent,
    EVENT_TEXT_CHUNK,
    EVENT_THINKING_CHUNK,
    EVENT_TOOL_CALL,
    EVENT_TOOL_CALL_UPDATE,
    EVENT_TOOL_RESULT,
    EVENT_PERMISSION_REQUEST,
    EVENT_COMPLETE,
    EVENT_CLEAR_STATUS,
    STOP_REASON_END_TURN,
    STOP_REASON_CANCELLED,
    OPTION_ALLOW_ONCE,
    OPTION_ALLOW_ALWAYS,
    TurnUsage,
)
from kiro_crew.providers.base import LLMProvider, CancelOutcome

from ._config import (
    DEFAULT_BASE_URL,
    DEFAULT_COMPACTION_KEEP_RECENT,
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    KNOWN_CONTEXT_WINDOWS,
    lookup_context_window,
    read_env_int,
    read_env_str,
)

logger = logging.getLogger(__name__)

# ── Tool execution registry ──────────────────────────────────────────────────
# Holds pending tool calls waiting for KiroCrew's approve/reject signal.
# key = request_id (str), value = asyncio.Future[bool]
_PENDING_APPROVALS: dict[str, asyncio.Future] = {}


class OpenAIProvider(LLMProvider):
    """LLMProvider backed by an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        context_window: int | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        tool_executor: "ToolExecutor | None" = None,
        session_key: str | None = None,
        compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
        compaction_keep_recent: int = DEFAULT_COMPACTION_KEEP_RECENT,
    ) -> None:
        self._base_url = base_url or read_env_str("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self._api_key = api_key or read_env_str("OPENAI_API_KEY")
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._tool_executor: ToolExecutor = tool_executor or DefaultToolExecutor()
        self._session_key = session_key

        # Context window for reporting (separate from max_tokens output limit)
        # If not provided, will be resolved from /v1/models in start()
        self._context_window_override = context_window  # explicit override (highest priority)
        self._context_window: int = context_window or 0  # final resolved value

        # Compaction settings
        self._compaction_threshold = compaction_threshold  # percent
        self._compaction_keep_recent = compaction_keep_recent  # messages to keep
        self._compacting: bool = False  # re-entrancy guard

        # Conversation history (maintained across turns)
        self._messages: list[dict[str, Any]] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

        # Context tracking
        self._context_pct: float = 0.0
        self._context_used: int = 0

        # Cancel support
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._alive: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._alive = True
        # Resolve context window from API if not explicitly overridden
        if not self._context_window_override:
            await self._resolve_context_window()
        logger.info(
            "OpenAIProvider started (model=%s, base_url=%s, context_window=%d, max_tokens=%d)",
            self._model, self._base_url, self._context_window, self._max_tokens,
        )

    async def _resolve_context_window(self) -> None:
        """Resolve the model's context window for usage reporting.

        Resolution priority:
          1. Constructor ``context_window`` param (already set in ``__init__``)
          2. ``OPENAI_CONTEXT_WINDOW`` env var
          3. ``/v1/models`` endpoint (``context_window`` / ``context_length``)
          4. Built-in ``KNOWN_CONTEXT_WINDOWS`` map (exact, then prefix)
          5. Fallback to ``max_tokens``
        """
        import httpx

        # Priority 1: explicit override (already set in __init__)
        if self._context_window > 0:
            return

        # Priority 2: env var
        env_ctx = read_env_int("OPENAI_CONTEXT_WINDOW")
        if env_ctx > 0:
            self._context_window = env_ctx
            logger.info(
                "Using context window from OPENAI_CONTEXT_WINDOW=%d",
                self._context_window,
            )
            return

        # Priority 3: query /v1/models
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                if resp.status_code == 200:
                    for m in resp.json().get("data", []):
                        if m.get("id") == self._model:
                            ctx = (
                                m.get("capabilities", {}).get("contextWindow")
                                or m.get("context_window")
                                or m.get("context_length")
                            )
                            if ctx and int(ctx) > 0:
                                self._context_window = int(ctx)
                                logger.info(
                                    "Resolved context window from /v1/models: "
                                    "%d (model=%s)",
                                    self._context_window, self._model,
                                )
                                return
                    logger.info(
                        "Model %s found in /v1/models but no context_window field",
                        self._model,
                    )
        except Exception as exc:
            logger.info("Could not query /v1/models: %s", exc)

        # Priority 4: built-in known models map
        known = lookup_context_window(self._model)
        if known is not None:
            self._context_window = known
            logger.info(
                "Using known context window for %s: %d",
                self._model, self._context_window,
            )
            return

        # Priority 5: fallback
        self._context_window = self._max_tokens
        logger.warning(
            "Could not determine context window for model=%s — "
            "falling back to max_tokens=%d. "
            "Set OPENAI_CONTEXT_WINDOW env var to override.",
            self._model, self._max_tokens,
        )

    async def shutdown(self) -> None:
        self._alive = False
        self._cancel_event.set()
        logger.info("OpenAIProvider shut down")

    def is_alive(self) -> bool:
        return self._alive

    # ── Context tracking ─────────────────────────────────────────────────────

    def context_usage_pct(self) -> float:
        return self._context_pct

    def context_used_tokens(self) -> int:
        return self._context_used

    def context_window_tokens(self) -> int:
        return self._context_window or self._max_tokens

    @property
    def served_model(self) -> str:
        return self._model

    # ── Auto-compaction ─────────────────────────────────────────────────────

    async def _maybe_compact(self) -> None:
        """Summarize old messages when context usage exceeds the threshold.

        When the context window is nearly full, this method:
          1. Preserves the system prompt and the last ``compaction_keep_recent``
             messages (recent conversation).
          2. Sends the middle messages to the same (or cheaper) model to produce
             a concise summary.
          3. Replaces the middle messages with a single summary message.

        The compaction is transparent to the user — a ``THINKING_CHUNK`` event
        is emitted so the dashboard shows "Auto-compacting…" in the thinking
        block, but no text content leaks into the main conversation.
        """
        # Guard: don't compact if already compacting or threshold is 0 (disabled)
        if self._compacting or self._compaction_threshold <= 0:
            return

        ctx_win = self._context_window or self._max_tokens
        if ctx_win <= 0:
            return

        # Estimate current usage: count tokens in all messages (rough: 4 chars ≈ 1 token)
        total_chars = sum(len(str(m.get("content", ""))) + len(str(m.get("tool_calls", ""))) for m in self._messages)
        estimated_tokens = total_chars // 4
        pct = min(estimated_tokens / ctx_win * 100, 100.0)

        if pct < self._compaction_threshold:
            return

        # Need compaction — figure out what to keep vs summarize
        # Always keep: system prompt (index 0 if present)
        has_system = self._messages and self._messages[0].get("role") == "system"
        offset = 1 if has_system else 0

        # Must have enough messages to compact
        if len(self._messages) < offset + self._compaction_keep_recent + 2:
            return

        keep_start = len(self._messages) - self._compaction_keep_recent
        old_messages = self._messages[offset:keep_start]
        if not old_messages:
            return

        self._compacting = True
        logger.info(
            "Auto-compaction triggered: %.1f%% context used (%d/%d tokens), "
            "compacting %d old messages, keeping %d recent",
            pct, estimated_tokens, ctx_win, len(old_messages), self._compaction_keep_recent,
        )

        try:
            summary = await self._summarize_for_compaction(old_messages)
            if not summary:
                logger.warning("Compaction summarization returned empty — skipping")
                return

            # Build compacted history: [system?] + [summary] + [recent messages]
            summary_msg = {
                "role": "system",
                "content": (
                    "[Auto-compacted conversation summary]\n"
                    "The following is a summary of earlier conversation messages "
                    "that were compressed to free up context space:\n\n"
                    f"{summary}"
                ),
            }
            new_messages: list[dict[str, Any]] = []
            if has_system:
                new_messages.append(self._messages[0])
            new_messages.append(summary_msg)
            new_messages.extend(self._messages[keep_start:])

            removed_count = len(self._messages) - len(new_messages)
            self._messages = new_messages
            logger.info(
                "Auto-compaction complete: %d messages removed, %d remaining "
                "(+1 summary)",
                removed_count, len(self._messages) - 1,
            )
        except Exception as exc:
            logger.warning("Auto-compaction failed: %s — continuing without compaction", exc)
        finally:
            self._compacting = False

    async def _summarize_for_compaction(self, messages: list[dict]) -> str:
        """Call the API (non-streaming) to summarize old messages."""
        import httpx

        # Build a summarization prompt
        # Flatten the messages into a readable transcript
        transcript_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")

            if role == "tool":
                # Tool result — include truncated
                transcript_parts.append(f"[Tool result]: {str(content)[:500]}")
            elif tool_calls:
                # Assistant tool-call turn
                names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                transcript_parts.append(f"[Assistant called tools: {', '.join(names)}]")
            elif content:
                # Normal text message
                transcript_parts.append(f"[{role}]: {str(content)[:1000]}")
            else:
                transcript_parts.append(f"[{role}]: (empty)")

        transcript = "\n".join(transcript_parts)

        summarization_messages = [
            {
                "role": "system",
                "content": (
                    "You are a conversation summarizer. Produce a concise but "
                    "complete summary of the conversation below. Preserve:\n"
                    "- Key decisions and conclusions\n"
                    "- Code changes made (file paths, what was changed)\n"
                    "- Tool calls and their outcomes\n"
                    "- User preferences expressed\n"
                    "- Open questions or pending tasks\n"
                    "Keep the summary under 2000 characters. Output ONLY the summary, "
                    "no preamble."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize this conversation:\n\n{transcript}",
            },
        ]

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": summarization_messages,
            "max_tokens": 800,
            "temperature": 0.1,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(
                    f"Compaction API error {resp.status_code}: {body.decode()[:300]}"
                )

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "")

    # ── Manual /compact command ─────────────────────────────────────────────

    async def compact(self, context: str = "") -> None:
        """Handle the /compact slash command.

        Summarizes the entire conversation history (except the system prompt)
        using the same summarization API, then replaces all messages with the
        summary.  This is the manual version of auto-compaction — triggered
        explicitly by the user typing ``/compact`` in the dashboard.
        """
        if self._compacting:
            logger.warning("compact() called while auto-compaction is in progress — skipping")
            return

        has_system = self._messages and self._messages[0].get("role") == "system"
        offset = 1 if has_system else 0

        # Nothing to compact
        if len(self._messages) <= offset + 1:
            logger.info("compact(): not enough messages to compact (%d)", len(self._messages))
            return

        # All messages except system prompt
        old_messages = self._messages[offset:]
        if not old_messages:
            return

        self._compacting = True
        logger.info("compact() triggered: summarizing %d messages", len(old_messages))

        try:
            summary = await self._summarize_for_compaction(old_messages)
            if not summary:
                logger.warning("compact(): summarization returned empty — skipping")
                return

            # Add user's context hint if provided
            if context:
                summary = f"{summary}\n\n[User-specified context to preserve: {context}]"

            summary_msg = {
                "role": "system",
                "content": (
                    "[Compacted conversation summary]\n"
                    "The full conversation history was compressed into this summary:\n\n"
                    f"{summary}"
                ),
            }

            new_messages: list[dict[str, Any]] = []
            if has_system:
                new_messages.append(self._messages[0])
            new_messages.append(summary_msg)

            removed_count = len(self._messages) - len(new_messages)
            self._messages = new_messages
            logger.info(
                "compact() complete: %d messages removed, %d remaining",
                removed_count, len(self._messages),
            )
        except Exception as exc:
            logger.warning("compact() failed: %s", exc)
        finally:
            self._compacting = False

    async def wait_for_compaction(self, timeout: float = 30.0) -> dict:
        """Wait for compaction to complete.

        Since OpenAI compaction is synchronous (happens within the compact() call),
        this returns immediately with the result status.
        """
        return {"type": "completed"}

    # ── Approval (tool permission) ────────────────────────────────────────────

    async def approve_tool(self, request_id: str | int, *, always: bool = False) -> None:
        key = str(request_id)
        fut = _PENDING_APPROVALS.get(key)
        if fut and not fut.done():
            fut.set_result(True)

    async def reject_tool(self, request_id: str | int) -> None:
        key = str(request_id)
        fut = _PENDING_APPROVALS.get(key)
        if fut and not fut.done():
            fut.set_result(False)

    # ── Cancel ───────────────────────────────────────────────────────────────

    async def cancel(self, *, wait_ack_timeout: float = 0.0) -> CancelOutcome:
        self._cancel_event.set()
        return "acked"

    # ── Main stream ──────────────────────────────────────────────────────────

    async def stream(self, message: str) -> AsyncIterator[AcpEvent]:  # type: ignore[override]
        """Send a user message and yield AcpEvents until end_turn or cancel."""
        self._cancel_event.clear()
        self._messages.append({"role": "user", "content": message})

        # Auto-compact if context usage is high
        await self._maybe_compact()

        # Tool definitions from executor
        tools = self._tool_executor.tool_definitions()

        async for event in self._run_turn(tools):
            if self._cancel_event.is_set():
                yield AcpEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_CANCELLED)
                return
            yield event

    async def _run_turn(self, tools: list[dict]) -> AsyncIterator[AcpEvent]:
        """Inner turn loop — handles tool call cycles.

        Text chunks are BUFFERED and only flushed to the UI once we know the
        finish_reason is 'stop' (not 'tool_calls').  This prevents the raw
        assistant preamble / tool-call XML from leaking into the dashboard
        when the model outputs text before deciding to call a tool.

        Thinking/reasoning chunks are forwarded immediately — they display in
        the collapsible thinking block and do not pollute the main message.
        """
        while True:
            tool_calls_made: list[dict] = []
            finish_reason = "stop"
            # Buffer for text chunks — flushed only when finish_reason != tool_calls
            text_buffer: list[AcpEvent] = []
            # Non-text, non-internal events (tool_call/result events from _stream_completion)
            other_events: list[AcpEvent] = []

            async for event in self._stream_completion(tools):
                if event.kind == "_tool_calls_collected":
                    tool_calls_made = json.loads(event.text)
                    continue
                if event.kind == "_finish_reason":
                    finish_reason = event.text
                    continue
                # Thinking chunks: forward immediately (goes to collapsible block, not main text)
                if event.kind == EVENT_THINKING_CHUNK:
                    yield event
                    continue
                # Text chunks: buffer — we don't know yet if this precedes a tool call
                if event.kind == EVENT_TEXT_CHUNK:
                    text_buffer.append(event)
                    continue
                # Everything else (rare in _stream_completion): buffer too
                other_events.append(event)

            if self._cancel_event.is_set():
                yield AcpEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_CANCELLED)
                return

            if finish_reason == "tool_calls" and tool_calls_made:
                # Discard text_buffer — it was preamble before the tool call, not real output.
                # Also check for text-format tool calls embedded in the buffer (some models).
                text_tool_calls = _extract_text_tool_calls(text_buffer)
                if text_tool_calls:
                    tool_calls_made = text_tool_calls

                # Flush any non-text events that should still be visible
                for ev in other_events:
                    yield ev

                # Execute each tool call, yield events, add results to history
                tool_results = []
                for tc in tool_calls_made:
                    async for ev in self._handle_tool_call(tc):
                        yield ev
                        if ev.kind == EVENT_TOOL_RESULT:
                            tool_results.append({
                                "tool_call_id": tc["id"],
                                "role": "tool",
                                "content": ev.tool_output,
                            })
                # Add assistant tool-call turn + results to history
                self._messages.append({
                    "role": "assistant",
                    "tool_calls": tool_calls_made,
                })
                self._messages.extend(tool_results)
                # Loop: send results back to model
                continue

            # --- finish_reason == "stop" (or other non-tool terminal) ---
            # Check if the text buffer contains text-format tool calls (fallback for models
            # that don't support structured tool_calls but output XML/JSON inline).
            text_tool_calls = _extract_text_tool_calls(text_buffer)
            if text_tool_calls:
                # Model used text-format tool calling — handle it the same way
                tool_calls_made = text_tool_calls
                for ev in other_events:
                    yield ev

                tool_results = []
                for tc in tool_calls_made:
                    async for ev in self._handle_tool_call(tc):
                        yield ev
                        if ev.kind == EVENT_TOOL_RESULT:
                            tool_results.append({
                                "tool_call_id": tc["id"],
                                "role": "tool",
                                "content": ev.tool_output,
                            })
                self._messages.append({"role": "assistant", "tool_calls": tool_calls_made})
                self._messages.extend(tool_results)
                continue

            # Normal text response — flush buffer to UI now
            for ev in text_buffer:
                yield ev
            for ev in other_events:
                yield ev

            yield AcpEvent(kind=EVENT_COMPLETE, stop_reason=STOP_REASON_END_TURN)
            return

    async def _stream_completion(self, tools: list[dict]) -> AsyncIterator[AcpEvent]:
        """Call the API with streaming and yield events."""
        import httpx  # deferred — only installed in openai_provider env

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._max_tokens:
            payload["max_tokens"] = self._max_tokens
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Accumulate tool calls across chunks
        accumulated_tool_calls: dict[int, dict] = {}
        finish_reason = "stop"
        assistant_text = ""

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise RuntimeError(
                        f"OpenAI API error {resp.status_code}: {body.decode()[:500]}"
                    )

                async for line in resp.aiter_lines():
                    if self._cancel_event.is_set():
                        return

                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # Usage (last chunk on many providers)
                    if usage := chunk.get("usage"):
                        pt = usage.get("prompt_tokens", 0)
                        ct = usage.get("completion_tokens", 0)
                        total = pt + ct
                        self._context_used = total
                        # Use resolved context_window for percentage calc (not max_tokens!)
                        ctx_win = self._context_window or self._max_tokens
                        if ctx_win:
                            self._context_pct = min(pt / ctx_win * 100, 100.0)

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr

                    # Thinking / reasoning tokens (o1, deepseek-r1, mimo, etc.)
                    # Field name varies by provider: reasoning_content (most), thinking (Anthropic-style)
                    thinking = delta.get("reasoning_content") or delta.get("thinking") or delta.get("reasoning")
                    if thinking:
                        yield AcpEvent(kind=EVENT_THINKING_CHUNK, text=thinking)

                    # Text content
                    content = delta.get("content") or ""
                    if content:
                        assistant_text += content
                        yield AcpEvent(kind=EVENT_TEXT_CHUNK, text=content)

                    # Tool call deltas (field may be null on some providers)
                    for tc_delta in (delta.get("tool_calls") or []):
                        idx = tc_delta.get("index", 0)
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        tc = accumulated_tool_calls[idx]
                        if tc_id := tc_delta.get("id"):
                            tc["id"] += tc_id
                        if fn := tc_delta.get("function", {}):
                            tc["function"]["name"] += fn.get("name") or ""
                            tc["function"]["arguments"] += fn.get("arguments") or ""

        # Add assistant text to history if no tool calls
        if assistant_text and not accumulated_tool_calls:
            self._messages.append({"role": "assistant", "content": assistant_text})

        # Signal collected tool calls via internal event
        if accumulated_tool_calls:
            tool_list = list(accumulated_tool_calls.values())
            yield AcpEvent(kind="_tool_calls_collected", text=json.dumps(tool_list))

        yield AcpEvent(kind="_finish_reason", text=finish_reason)

    async def _handle_tool_call(self, tc: dict) -> AsyncIterator[AcpEvent]:
        """Emit permission_request → (approved?) → execute → tool_result."""
        fn_name = tc["function"]["name"]
        try:
            fn_args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            fn_args = {}

        tool_call_id = tc.get("id") or str(uuid.uuid4())
        request_id = str(uuid.uuid4())

        # Announce tool call to dashboard
        purpose = fn_args.pop("__tool_use_purpose", "") or fn_name
        yield AcpEvent(
            kind=EVENT_TOOL_CALL,
            tool_call_id=tool_call_id,
            title=purpose,
            tool_name=fn_name,
            tool_purpose=purpose,
            tool_input=json.dumps(fn_args),
        )

        # Request permission from KiroCrew approval ladder
        fut: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        _PENDING_APPROVALS[request_id] = fut

        yield AcpEvent(
            kind=EVENT_PERMISSION_REQUEST,
            request_id=request_id,
            title=purpose,
            tool_call_id=tool_call_id,
            tool_name=fn_name,
            tool_input=json.dumps(fn_args),
            options=[
                {"id": OPTION_ALLOW_ONCE, "label": "Allow once"},
                {"id": OPTION_ALLOW_ALWAYS, "label": "Always allow"},
            ],
        )

        try:
            approved = await asyncio.wait_for(fut, timeout=120.0)
        except asyncio.TimeoutError:
            approved = False
        finally:
            _PENDING_APPROVALS.pop(request_id, None)

        if not approved:
            yield AcpEvent(
                kind=EVENT_TOOL_RESULT,
                tool_call_id=tool_call_id,
                tool_output="[Tool call rejected by user]",
                tool_final=True,
            )
            return

        # Execute the tool
        try:
            result = await self._tool_executor.execute(fn_name, fn_args)
        except Exception as exc:
            result = f"[Tool error: {exc}]"
            logger.exception("Tool %s raised", fn_name)

        result_str = result if isinstance(result, str) else json.dumps(result)

        yield AcpEvent(
            kind=EVENT_TOOL_RESULT,
            tool_call_id=tool_call_id,
            tool_output=result_str,
            tool_final=True,
        )


# ── Text-format tool call parser ─────────────────────────────────────────────


def _extract_text_tool_calls(text_buffer: list) -> list[dict]:
    """Parse text-format tool calls from buffered TEXT_CHUNK events.

    Some models (especially when given a long system prompt) emit tool calls as
    structured text rather than structured tool_calls JSON.  We support several
    common formats:

    1. KiroCrew XML style (from system prompt examples):
       <tool_call>
         <function=bash>
           <parameter=command>ls -la</parameter>
         </function>
       </tool_call>

    2. Hermes/Nous format:
       <tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>

    3. Plain JSON block:
       ```json
       {"name": "bash", "arguments": {"command": "ls"}}
       ```

    Returns a list of OpenAI-format tool call dicts, or [] if none found.
    """
    if not text_buffer:
        return []

    # Reconstruct full text from buffer
    full_text = "".join(ev.text for ev in text_buffer if hasattr(ev, "text") and ev.text)
    if not full_text.strip():
        return []

    tool_calls = []

    # ── Format 1: KiroCrew XML style ─────────────────────────────────────────
    # <tool_call>\n  <function=NAME>\n    <parameter=KEY>VALUE</parameter>\n  </function>\n</tool_call>
    xml_blocks = _re.findall(r"<tool_call>(.*?)</tool_call>", full_text, _re.DOTALL)
    for block in xml_blocks:
        # Try JSON first (Hermes format inside <tool_call>)
        block_stripped = block.strip()
        if block_stripped.startswith("{"):
            try:
                obj = json.loads(block_stripped)
                name = obj.get("name") or obj.get("function", {}).get("name", "")
                args = obj.get("arguments") or obj.get("parameters") or obj.get("input") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"input": args}
                if name:
                    tool_calls.append({
                        "id": f"tc_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    })
                    continue
            except json.JSONDecodeError:
                pass

        # KiroCrew XML: <function=NAME> <parameter=KEY>VALUE</parameter> ...
        fn_match = _re.search(r"<function=([^>]+)>", block)
        if fn_match:
            fn_name = fn_match.group(1).strip()
            params = {}
            for pm in _re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", block, _re.DOTALL):
                params[pm.group(1).strip()] = pm.group(2).strip()
            tool_calls.append({
                "id": f"tc_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": fn_name, "arguments": json.dumps(params)},
            })

    if tool_calls:
        return tool_calls

    # ── Format 2: JSON code block ─────────────────────────────────────────────
    json_blocks = _re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", full_text, _re.DOTALL)
    for block in json_blocks:
        try:
            obj = json.loads(block)
            name = obj.get("name") or obj.get("tool") or obj.get("function", {}).get("name", "")
            args = obj.get("arguments") or obj.get("parameters") or obj.get("input") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"input": args}
            if name:
                tool_calls.append({
                    "id": f"tc_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                })
        except json.JSONDecodeError:
            continue

    return tool_calls


# ── Tool Executor interface ───────────────────────────────────────────────────

class ToolExecutor:
    """Override this to plug in real tool execution (MCP, bash, file I/O)."""

    def tool_definitions(self) -> list[dict]:
        """Return OpenAI-format tool definitions."""
        return []

    async def execute(self, name: str, args: dict) -> Any:
        raise NotImplementedError(f"No executor for tool: {name}")


class DefaultToolExecutor(ToolExecutor):
    """No-op executor — no tools available."""

    def tool_definitions(self) -> list[dict]:
        return []
