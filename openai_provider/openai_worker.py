"""OpenAIWorker — Knowledge extraction worker via OpenAI-compatible API.

Drop-in replacement for AcpWorker when ``agent.provider=openai``.
Uses non-streaming chat completions (extraction prompts are self-contained,
no streaming needed — cheaper and simpler).

Patches ``LLMPool._create_worker`` at runtime so no installed-package files
are modified.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ._config import (
    KNOWN_CONTEXT_WINDOWS,
    lookup_context_window,
    read_knowledge_config,
)

if TYPE_CHECKING:
    from kiro_crew.knowledge.llm_pool import Worker

logger = logging.getLogger(__name__)

# Lazy import guard — httpx is in the kirocrew venv but may not be importable
# at module level in all contexts (tests, standalone scripts).
_httpx: Any = None


def _ensure_httpx() -> Any:
    """Import and cache ``httpx`` on first use."""
    global _httpx
    if _httpx is None:
        import httpx as _mod
        _httpx = _mod
    return _httpx


# ── OpenAIWorker ─────────────────────────────────────────────────────────────


class OpenAIWorker:
    """Long-lived worker backed by an OpenAI-compatible chat-completions API.

    Implements the same duck-typed interface as
    ``kiro_crew.knowledge.llm_pool.Worker`` (``start``, ``send_message``,
    ``shutdown``, ``is_alive``, ``reset_conversation``, ``context_pct``) so it
    can be used by ``LLMPool`` without modification.

    Unlike ``AcpClient`` / ``CCWorker``, this worker does **not** spawn a
    subprocess — it's a pure HTTP client, lighter on resources.
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._client: Any = None  # httpx.AsyncClient
        self._alive: bool = False
        self._messages: list[dict[str, Any]] = []
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._context_window: int = 0
        self.calls_since_reset: int = 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise the HTTP client and resolve the context window."""
        httpx = _ensure_httpx()
        self._config = read_knowledge_config()

        if not self._config["api_key"]:
            raise RuntimeError(
                "OpenAIWorker: OPENAI_API_KEY not set. "
                "Set it in environment or config.json."
            )

        # Close existing client if any (needed for reset_conversation)
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass

        self._client = httpx.AsyncClient(
            base_url=self._config["base_url"],
            headers={
                "Authorization": f"Bearer {self._config['api_key']}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        )
        self._messages = []
        self._alive = True
        self._prompt_tokens = 0
        self._completion_tokens = 0

        await self._resolve_context_window()

        logger.info(
            "OpenAIWorker: started (model=%s, base_url=%s, context_window=%d)",
            self._config["model"],
            self._config["base_url"],
            self._context_window,
        )

    async def shutdown(self) -> None:
        """Close the HTTP client and mark the worker as dead."""
        self._alive = False
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                logger.debug("OpenAIWorker: client close error", exc_info=True)
            self._client = None
        self._messages = []
        logger.info("OpenAIWorker: shut down")

    def is_alive(self) -> bool:
        """``True`` if the HTTP client is still usable."""
        return self._alive and self._client is not None

    # ── Context window resolution ────────────────────────────────────────────

    async def _resolve_context_window(self) -> None:
        """Resolve the model's context window for usage reporting.

        Resolution order:
          1. ``/v1/models`` endpoint (``context_window`` / ``context_length``)
          2. Built-in ``KNOWN_CONTEXT_WINDOWS`` map (exact, then prefix match)
          3. Fallback to ``max_tokens``
        """
        if self._context_window > 0:
            return

        # Try /v1/models
        try:
            resp = await self._client.get("/models")
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("data", []):
                    if m.get("id") == self._config["model"]:
                        ctx = (
                            m.get("capabilities", {}).get("contextWindow")
                            or m.get("context_window")
                            or m.get("context_length")
                        )
                        if ctx and int(ctx) > 0:
                            self._context_window = int(ctx)
                            return
        except Exception:
            pass

        # Known models map
        known = lookup_context_window(self._config["model"])
        if known is not None:
            self._context_window = known
            return

        # Fallback
        self._context_window = self._config["max_tokens"]

    # ── Messaging ────────────────────────────────────────────────────────────

    async def send_message(self, prompt: str, timeout: float = 60.0) -> str:
        """Send a prompt via non-streaming chat completion.

        Returns the assistant's response text.  Accumulates messages for
        multi-turn, but knowledge extraction prompts are self-contained so
        this is effectively stateless.
        """
        if not self._alive or self._client is None:
            await self.start()

        self._messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._config["model"],
            "messages": self._messages,
            "max_tokens": self._config["max_tokens"],
            "temperature": 0.0,
            "stream": False,
        }

        try:
            resp = await self._client.post(
                "/chat/completions",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("OpenAIWorker: API call failed: %s", exc)
            # Roll back the user message we just added
            if self._messages and self._messages[-1]["role"] == "user":
                self._messages.pop()
            raise

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"OpenAIWorker: empty response: {json.dumps(data)[:500]}"
            )

        text = choices[0].get("message", {}).get("content", "")

        # Track usage
        usage = data.get("usage", {})
        self._prompt_tokens += usage.get("prompt_tokens", 0)
        self._completion_tokens += usage.get("completion_tokens", 0)

        # Append assistant response to conversation history
        self._messages.append({"role": "assistant", "content": text})

        return text

    # ── Context tracking ─────────────────────────────────────────────────────

    def context_pct(self) -> float:
        """Percentage of context window used (``0.0`` if window unknown)."""
        if self._context_window <= 0:
            return 0.0
        return min(100.0, (self._prompt_tokens / self._context_window) * 100.0)

    async def reset_conversation(self) -> None:
        """Drop the accumulated transcript, keeping the HTTP client alive.

        Unlike ``AcpWorker`` / ``CCWorker`` there's no subprocess to respawn —
        just clear the message history.
        """
        self._messages = []
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self.calls_since_reset = 0
        logger.info("OpenAIWorker: conversation reset")


# ── Registration (monkey-patch) ──────────────────────────────────────────────


def _register_openai_worker() -> None:
    """Monkey-patch ``LLMPool._create_worker`` to use ``OpenAIWorker``.

    Called by ``install.py`` after patching the main provider.
    Does **not** modify any installed-package files — pure runtime patch.
    """
    try:
        from kiro_crew.knowledge.llm_pool import LLMPool

        _orig_create_worker = LLMPool._create_worker

        async def _patched_create_worker(self: Any) -> Any:
            """Use ``OpenAIWorker`` when ``provider_type`` is ``'openai'``."""
            if self._provider_type == "openai":
                worker = OpenAIWorker()
                await worker.start()
                return worker
            return await _orig_create_worker(self)

        LLMPool._create_worker = _patched_create_worker  # type: ignore[method-assign]
        logger.info("OpenAIWorker: LLMPool._create_worker monkey-patched ✅")

    except ImportError:
        logger.warning(
            "OpenAIWorker: could not import LLMPool. "
            "Knowledge extraction will fall back to ACP."
        )
    except Exception as exc:
        logger.warning("OpenAIWorker: patch failed: %s", exc)
