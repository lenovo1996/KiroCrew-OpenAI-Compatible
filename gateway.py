#!/usr/bin/env python3
"""gateway.py — Start KiroCrew gateway with OpenAI-compatible provider.

Usage:
    OPENAI_BASE_URL=http://localhost:11434/v1 \
    OPENAI_API_KEY=ollama \
    OPENAI_MODEL=llama3.1:70b \
    python gateway.py

    # Or for standard OpenAI:
    OPENAI_API_KEY=sk-... python gateway.py

    # For vLLM / LiteLLM proxy:
    OPENAI_BASE_URL=http://localhost:8000/v1 \
    OPENAI_MODEL=mistral-7b-instruct \
    python gateway.py
"""

import sys
import os
import logging

# ── Encoding fix ──────────────────────────────────────────────────────────────
# Force UTF-8 on stdout/stderr and all subprocess pipes.
# Without this, Python defaults to the locale encoding (often ASCII on servers),
# causing UnicodeEncodeError when model responses contain non-ASCII chars (…, —, etc.)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")  # Python 3.7+ UTF-8 mode

# Reconfigure stdout/stderr in-process for the current interpreter
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Ensure openai_provider package is importable — must come FIRST.
# Adds the repo root (directory containing this file) to sys.path so that
# ``import openai_provider`` resolves to the ``openai_provider/`` subdirectory
# regardless of where the script is invoked from.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# 1. Install the provider patch BEFORE importing anything from kiro_crew
import openai_provider
openai_provider.install()

# 2. Inject 'gateway' subcommand if not already present
if len(sys.argv) == 1 or sys.argv[1] != "gateway":
    sys.argv.insert(1, "gateway")

# 3. Now boot the KiroCrew gateway normally
from kiro_crew.cli import main
main()
