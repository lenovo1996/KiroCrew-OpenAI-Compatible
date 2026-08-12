"""install.py — Monkey-patch KiroCrew to use OpenAIProvider.

Call ``install()`` **before** ``kirocrew gateway`` initialises its provider
factory.  The patch replaces the ``ProviderRegistry.create_factory`` seam so
every new session spawns an ``OpenAIProvider`` instead of ``AcpProvider``.

Also patches ``LLMPool._create_worker`` so knowledge extraction uses
``OpenAIWorker`` (pure HTTP) instead of ``AcpClient`` (kiro-cli subprocess).

Environment variables (read at ``install()`` time):

    OPENAI_BASE_URL         API endpoint (default: ``https://api.openai.com/v1``)
    OPENAI_API_KEY          API key (required)
    OPENAI_MODEL            model name (default: ``gpt-4o``)
    OPENAI_MAX_TOKENS       max output tokens per turn (default: ``8192``)
    OPENAI_CONTEXT_WINDOW   model context window for usage reporting
                            (default: auto-resolve from API)
    OPENAI_SYSTEM_PROMPT    override system prompt (optional)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ._config import read_provider_config

logger = logging.getLogger(__name__)

_installed = False

# KiroCrew sentinel values that mean "no model pinned, pick the default".
# When the session manager passes one of these as model_override, the factory
# must fall through to the configured model — otherwise the remote API proxy
# receives "auto" (or similar) and returns 404 "model_not_found".
_MODEL_SENTINELS = frozenset({"auto", ""})


def install(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    tool_executor: Any | None = None,
) -> None:
    """Patch KiroCrew's ``ProviderRegistry`` to return ``OpenAIProvider``.

    Safe to call multiple times — subsequent calls are no-ops.
    Explicit keyword arguments override environment variables.
    """
    global _installed
    if _installed:
        return

    from .provider import OpenAIProvider

    # Merge explicit args with env-var config (explicit wins).
    cfg = read_provider_config()
    cfg["base_url"] = base_url or cfg["base_url"]
    cfg["api_key"] = api_key or cfg["api_key"]
    cfg["model"] = model or cfg["model"]
    cfg["max_tokens"] = max_tokens or cfg["max_tokens"]
    cfg["system_prompt"] = system_prompt or cfg["system_prompt"]
    # context_window: explicit param > env var (already in cfg)
    if cfg["context_window"] is None:
        cfg["context_window"] = None  # will auto-resolve in provider

    if not cfg["api_key"]:
        logger.warning(
            "openai_provider: OPENAI_API_KEY not set — provider will fail on first request"
        )

    def _openai_factory(
        session_key: str | None = None,
        agent: str | None = None,
        channel_id: str | None = None,
        model_override: str | None = None,
        cwd: str | None = None,
        extra_env: dict | None = None,
        reasoning_effort_override: str | None = None,
        **_kwargs: object,
    ) -> OpenAIProvider:
        # Resolve model: explicit override wins unless it's a sentinel.
        if model_override and model_override not in _MODEL_SENTINELS:
            used_model = model_override
        else:
            used_model = cfg["model"]
            if model_override:
                logger.info(
                    "openai_provider: model_override=%r is a sentinel — "
                    "falling back to configured model=%s",
                    model_override, cfg["model"],
                )
        executor = tool_executor or _build_mcp_executor()
        return OpenAIProvider(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=used_model,
            system_prompt=cfg["system_prompt"],
            max_tokens=cfg["max_tokens"],
            context_window=cfg["context_window"],
            tool_executor=executor,
            session_key=session_key,
        )

    # ── Patch 1: build_provider_factory (module-level function) ──────────────
    try:
        from kiro_crew.config import loader as _loader_mod

        def _patched_build(_cfg: Any) -> Callable:
            logger.info(
                "openai_provider: intercepted build_provider_factory — "
                "returning OpenAI factory"
            )
            return _openai_factory

        _loader_mod.build_provider_factory = _patched_build  # type: ignore[attr-defined]
        logger.info("openai_provider: build_provider_factory patched ✅")

    except Exception as exc:
        logger.error("openai_provider: patch failed: %s", exc)
        return

    # ── Patch 2: KiroCrewConfig.create_provider_factory (instance method) ───
    try:
        from kiro_crew.config.loader import KiroCrewConfig

        KiroCrewConfig.create_provider_factory = lambda self: _openai_factory  # type: ignore[method-assign]
        logger.info("openai_provider: KiroCrewConfig.create_provider_factory patched ✅")
    except Exception as exc:
        logger.warning("openai_provider: KiroCrewConfig patch skipped: %s", exc)

    # ── Patch 3: LLMPool._create_worker (knowledge extraction) ──────────────
    try:
        from .openai_worker import _register_openai_worker
        _register_openai_worker()
    except Exception as exc:
        logger.warning(
            "openai_provider: knowledge worker registration skipped: %s", exc
        )

    _installed = True
    logger.info(
        "openai_provider installed — model=%s  base_url=%s",
        cfg["model"], cfg["base_url"],
    )


def _build_mcp_executor() -> Any:
    """Build a ``ToolExecutor`` backed by KiroCrew's MCP infrastructure."""
    try:
        from .mcp_executor import McpToolExecutor
        return McpToolExecutor()
    except Exception:
        from .provider import DefaultToolExecutor
        return DefaultToolExecutor()
