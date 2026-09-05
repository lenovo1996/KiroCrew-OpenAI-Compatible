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

# Timeout for upstream /v1/models HTTP requests (seconds).
_UPSTREAM_MODELS_TIMEOUT = 10.0


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
            compaction_threshold=cfg.get("compaction_threshold", 80),
            compaction_keep_recent=cfg.get("compaction_keep_recent", 6),
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
        logger.warning("openai_provider: build_provider_factory patch failed: %s", exc)
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

    # ── Patch 4: Embedding backend (replace bundled llama.cpp) ─────────────
    try:
        from .embedding_backend import register_remote_embedding
        register_remote_embedding()
    except Exception as exc:
        logger.warning(
            "openai_provider: embedding backend registration skipped: %s", exc
        )

    # ── Patch 5: Background sessions (auto-title, link summary, etc.) ─────
    # ``run_bg_oneliner`` (used by chat_title, chat_nav) calls
    # ``sessions.get_bg_session()`` which checks ``_bg_provider_is_kiro()``.
    # That method reads ``agent.provider`` from config.json — if it's "acp",
    # bg sessions route to AcpSessionHandle → kiro-cli → Anthropic, bypassing
    # our patched factory entirely.  Overriding it to return False makes
    # ``get_bg_session`` fall through to ``_ensure_background()`` which
    # creates a session via the (already-patched) provider factory →
    # OpenAIProvider → 9router.  The ``_ProviderBgSession`` wrapper provides
    # the same ``prompt()``/``reject_tool()``/``destroy()`` interface that
    # ``run_bg_oneliner`` expects.
    try:
        from kiro_crew.session import SessionManager

        def _bg_provider_is_not_kiro(self: Any) -> bool:
            return False

        SessionManager._bg_provider_is_kiro = _bg_provider_is_not_kiro  # type: ignore[method-assign]
        logger.info(
            "openai_provider: SessionManager._bg_provider_is_kiro patched → "
            "bg sessions will use OpenAI factory ✅"
        )
    except Exception as exc:
        logger.warning(
            "openai_provider: bg session patch skipped: %s", exc
        )

    # ── Patch 6: /api/models — upstream-first catalog ───────────────────
    # Replaces the stock /api/models handler (which shells out to
    # ``kiro-cli --list-models``) with one that queries the upstream
    # OpenAI-compatible ``/v1/models`` endpoint first.  The kiro-cli
    # handler is kept as a graceful fallback if upstream is unreachable.
    try:
        _patch_api_models_fallback(cfg["base_url"], cfg["api_key"])
    except Exception as exc:
        logger.warning("openai_provider: /api/models fallback patch skipped: %s", exc)

    _installed = True
    logger.info(
        "openai_provider installed — model=%s  base_url=%s",
        cfg["model"], cfg["base_url"],
    )


async def _fetch_upstream_models(
    base_url: str, api_key: str,
) -> list[dict[str, object]] | None:
    """Fetch and format models from the upstream OpenAI-compatible ``/v1/models``.

    Returns a list of model dicts formatted for the KiroCrew frontend
    (``{model_name, description, context_window_tokens, rate_multiplier}``),
    or ``None`` if the upstream is unreachable, returned an error, or served
    zero models.

    Uses ``model_registry.model_window()`` for context window resolution,
    falling back to ``0`` for unknown models.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=_UPSTREAM_MODELS_TIMEOUT) as client:
            resp = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code != 200:
                logger.warning(
                    "openai_provider: upstream /models returned %d", resp.status_code,
                )
                return None

            raw_models = resp.json().get("data", [])
    except Exception:
        logger.warning("openai_provider: upstream /models query failed", exc_info=True)
        return None

    from kiro_crew import model_registry

    formatted = [
        {
            "model_name": raw["id"],
            "description": "",
            "context_window_tokens": model_registry.model_window(raw["id"]) or 0,
            "rate_multiplier": 1.0,
        }
        for raw in raw_models
        if raw.get("id")
    ]

    if not formatted:
        logger.warning("openai_provider: upstream /models returned 0 models")
        return None

    logger.info(
        "openai_provider: upstream /models returned %d models from %s",
        len(formatted), base_url,
    )
    return formatted


def _patch_api_models_fallback(base_url: str, api_key: str) -> None:
    """Replace ``/api/models`` with an upstream-first model catalog.

    Queries the upstream OpenAI-compatible ``/v1/models`` endpoint.  If
    unavailable, falls back to the stock ``kiro-cli --list-models`` handler.
    If both fail, returns a 503.

    Also patches the ``handlers`` package-level binding so the aiohttp
    router (which resolves via ``handlers.api_models``, not
    ``handlers.agents.api_models``) picks up the replacement.
    """
    from kiro_crew.dashboard.handlers import agents as _agents_mod

    _original_handler = _agents_mod.api_models

    async def _api_models_with_upstream(request):  # type: ignore[no-untyped-def]
        """Upstream-first /api/models with kiro-cli fallback."""
        from aiohttp import web

        # Try upstream first — it's the authoritative source.
        models = await _fetch_upstream_models(base_url, api_key)
        if models is not None:
            return web.json_response(models)

        # Upstream unavailable — fall back to stock kiro-cli handler.
        logger.info("openai_provider: falling back to stock kiro-cli /api/models")
        try:
            return await _original_handler(request)
        except Exception:
            logger.warning(
                "openai_provider: stock /api/models also failed", exc_info=True,
            )
            return web.json_response(
                {"error": "model list unavailable"}, status=503,
            )

    # Patch both the module-level and package-level bindings — ``server.py``
    # routes via ``handlers.api_models`` (the package __init__ re-export),
    # not ``handlers.agents.api_models``.
    _agents_mod.api_models = _api_models_with_upstream  # type: ignore[assignment]
    import kiro_crew.dashboard.handlers as _handlers_pkg
    _handlers_pkg.api_models = _api_models_with_upstream  # type: ignore[assignment]

    logger.info("openai_provider: /api/models replaced with upstream catalog ✅")


def _build_mcp_executor() -> Any:
    """Build a ``ToolExecutor`` backed by KiroCrew's MCP infrastructure."""
    try:
        from .mcp_executor import McpToolExecutor
        return McpToolExecutor()
    except Exception:
        from .provider import DefaultToolExecutor
        return DefaultToolExecutor()
