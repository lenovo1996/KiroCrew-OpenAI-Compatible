"""OpenAIEmbeddingBackend — remote embedding via OpenAI-compatible API.

Drop-in replacement for the bundled LlamaCppEmbedder.  Calls an external
``POST /v1/embeddings`` endpoint instead of loading the ~610 MB GGUF model
in-process, saving both RAM (~700 MB RSS) and CPU (no local matrix-multiply).

Designed to be registered via ``kiro_crew.embeddings.register_embedding_backend``
so that knowledge ingestion and vector memory use it transparently.

Configuration (environment variables, read at import time):

    EMBEDDING_BASE_URL      API endpoint (default: same as OPENAI_BASE_URL)
    EMBEDDING_API_KEY       Auth key  (default: same as OPENAI_API_KEY)
    EMBEDDING_MODEL         Model id  (default: ``nvidia/nvidia/nemotron-3-embed-1b``)
    EMBEDDING_DIM           Vector dimensionality (default: 2048)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from kiro_crew.embeddings import EmbeddingBackend

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_BASE_URL = "http://localhost:20128/v1"
_DEFAULT_API_KEY = ""
_DEFAULT_MODEL = "nvidia/nvidia/nemotron-3-embed-1b"
_DEFAULT_DIM = 2048
_TIMEOUT_SECS = 30.0
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds, doubles each retry


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# ── Backend ──────────────────────────────────────────────────────────────────


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Embedding backend that calls an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Thread-safe: urllib.request is inherently thread-safe, and we use no shared
    mutable state beyond the read-only config.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dim: int | None = None,
    ) -> None:
        self._base_url = (base_url or _env("EMBEDDING_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or _env("EMBEDDING_API_KEY") or _env("OPENAI_API_KEY") or _DEFAULT_API_KEY
        self._model = model or _env("EMBEDDING_MODEL") or _DEFAULT_MODEL
        self._dim = dim or int(_env("EMBEDDING_DIM") or _DEFAULT_DIM)
        self._closed = False
        self._endpoint_url = f"{self._base_url}/embeddings"

        logger.info(
            "OpenAIEmbeddingBackend created (model=%s, dim=%d, url=%s)",
            self._model, self._dim, self._endpoint_url,
        )

    # ── EmbeddingBackend ABC ────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def is_ready(self) -> bool:
        """Ready when the endpoint is reachable (cheap HEAD or model-list probe)."""
        if self._closed:
            return False
        try:
            req = urllib.request.Request(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float] | None:
        """Embed a single text. Returns None on any failure."""
        result = self.embed_batch([text])
        return result[0] if result else None

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Embed multiple texts via the remote API.

        Returns None on any failure (network, API error, dimension mismatch).
        Retries transient errors with exponential backoff.
        """
        if not texts or not any(t.strip() for t in texts):
            return None
        if self._closed:
            return None

        # Truncate very long inputs (matching LlamaCppEmbedder's _MAX_EMBED_CHARS)
        clipped = [t[:6000] if len(t) > 6000 else t for t in texts]

        payload = json.dumps({
            "model": self._model,
            "input": clipped,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    self._endpoint_url,
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=_TIMEOUT_SECS) as resp:
                    body = resp.read()

                data = json.loads(body)
                entries = data.get("data", [])
                if not entries:
                    logger.warning("Embedding API returned empty data")
                    return None

                # Sort by index to preserve ordering
                entries.sort(key=lambda e: e.get("index", 0))
                vectors = [e["embedding"] for e in entries]

                # Validate
                if len(vectors) != len(clipped):
                    logger.warning(
                        "Embedding count mismatch: got %d, expected %d",
                        len(vectors), len(clipped),
                    )
                    return None

                for i, v in enumerate(vectors):
                    if len(v) != self._dim:
                        logger.warning(
                            "Embedding dim mismatch: vector %d has %d dims, expected %d",
                            i, len(v), self._dim,
                        )
                        return None

                return vectors

            except urllib.error.HTTPError as exc:
                body_text = ""
                try:
                    body_text = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                logger.warning(
                    "Embedding API HTTP %d (attempt %d/%d): %s",
                    exc.code, attempt + 1, _MAX_RETRIES, body_text,
                )
                # Don't retry 4xx (client errors)
                if 400 <= exc.code < 500:
                    return None
                last_error = exc

            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                logger.warning(
                    "Embedding API network error (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, exc,
                )
                last_error = exc

            except Exception as exc:
                logger.warning(
                    "Embedding API unexpected error (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, exc,
                )
                last_error = exc

            # Exponential backoff before retry
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
                time.sleep(delay)

        logger.error("Embedding API failed after %d attempts: %s", _MAX_RETRIES, last_error)
        return None

    def close(self) -> None:
        """Release resources (no-op for HTTP backend, but marks as closed)."""
        self._closed = True
        logger.info("OpenAIEmbeddingBackend closed")


# ── Registration helper ──────────────────────────────────────────────────────


def _make_openai_embedding_backend() -> OpenAIEmbeddingBackend:
    """Factory function for register_embedding_backend()."""
    return OpenAIEmbeddingBackend()


def register_remote_embedding() -> None:
    """Register the remote embedding backend with KiroCrew.

    Call this from install() to replace the bundled llama.cpp embedder.
    Also resets the shared embedder singleton so the new backend takes effect
    immediately.
    """
    try:
        from kiro_crew.embeddings import register_embedding_backend, reset_shared_embedder

        register_embedding_backend(_make_openai_embedding_backend)
        reset_shared_embedder()
        logger.info("openai_provider: remote embedding backend registered ✅")
    except Exception as exc:
        logger.warning("openai_provider: embedding backend registration skipped: %s", exc)
