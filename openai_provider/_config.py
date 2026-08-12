"""_config.py — Shared configuration and constants for openai_provider.

Centralises environment variable reading, default values, and the
known-models context-window map so that provider.py, openai_worker.py,
and install.py all draw from a single source of truth.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0

# ── Known context windows ────────────────────────────────────────────────────
# Used as a fallback when the /v1/models endpoint is unavailable or does not
# report context_window for a given model.  Keyed by model id; prefix
# matching is applied after exact match.

KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    "free": 1_048_576,                  # MiMo-v2.5-pro (1M)
    "mimo-v2.5-pro": 1_048_576,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 128_000,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-sonnet-4.5": 200_000,
    "claude-haiku-4-20250506": 200_000,
    "deepseek-r1": 65_536,
    "deepseek-chat": 65_536,
    "qwen3-coder-next": 1_048_576,
}


def lookup_context_window(model: str) -> int | None:
    """Return the known context window for *model*, or ``None``.

    Tries an exact key match first, then falls back to prefix matching.
    """
    if model in KNOWN_CONTEXT_WINDOWS:
        return KNOWN_CONTEXT_WINDOWS[model]
    for prefix, ctx in KNOWN_CONTEXT_WINDOWS.items():
        if model.startswith(prefix):
            return ctx
    return None


# ── Environment variable helpers ─────────────────────────────────────────────

def read_env_str(name: str, default: str = "") -> str:
    """Read a string environment variable, returning *default* if unset."""
    return os.environ.get(name, default)


def read_env_int(name: str, default: int = 0) -> int:
    """Read an integer environment variable, returning *default* on error."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using default %d", name, raw, default)
        return default


# ── Provider config (for install.py / main chat sessions) ───────────────────

def read_provider_config() -> dict[str, Any]:
    """Read provider configuration from environment variables.

    Returns a dict with keys: ``base_url``, ``api_key``, ``model``,
    ``max_tokens``, ``context_window``, ``system_prompt``.
    """
    return {
        "base_url": read_env_str("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        "api_key": read_env_str("OPENAI_API_KEY"),
        "model": read_env_str("OPENAI_MODEL", DEFAULT_MODEL),
        "max_tokens": read_env_int("OPENAI_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        "context_window": read_env_int("OPENAI_CONTEXT_WINDOW") or None,
        "system_prompt": read_env_str("OPENAI_SYSTEM_PROMPT"),
    }


# ── Knowledge worker config ─────────────────────────────────────────────────

def read_knowledge_config() -> dict[str, Any]:
    """Read knowledge-worker configuration from env vars + config.json.

    The knowledge worker can use a separate model (``OPENAI_KNOWLEDGE_MODEL``)
    for cheaper/faster background extraction.  Falls back to ``OPENAI_MODEL``.

    Also reads ``knowledge.openai_model`` and ``knowledge.openai_max_tokens``
    from the KiroCrew config.json if present.
    """
    model = (
        read_env_str("OPENAI_KNOWLEDGE_MODEL")
        or read_env_str("OPENAI_MODEL", DEFAULT_MODEL)
    )
    max_tokens = read_env_int("OPENAI_MAX_TOKENS", DEFAULT_MAX_TOKENS)

    # Config.json overrides (knowledge-specific)
    try:
        from kiro_crew.config.paths import config_dir
        config_path = config_dir() / "config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text())
            if isinstance(data, dict):
                knowledge = data.get("knowledge")
                if isinstance(knowledge, dict):
                    model = knowledge.get("openai_model", model)
                    max_tokens = knowledge.get("openai_max_tokens", max_tokens)
    except Exception:
        pass

    return {
        "base_url": read_env_str("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        "api_key": read_env_str("OPENAI_API_KEY"),
        "model": model,
        "max_tokens": max_tokens,
    }
