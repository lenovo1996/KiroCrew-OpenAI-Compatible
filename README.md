# KiroCrew-OpenAI-Compatible

**Drop-in OpenAI-compatible backend for [KiroCrew](https://github.com/kirocrew/kirocrew).**

<img width="1918" height="911" alt="image" src="https://github.com/user-attachments/assets/7e5288dd-4d75-48eb-b70a-6dd8608d08c5" />

Run KiroCrew with any OpenAI-compatible API — Ollama, vLLM, LiteLLM, Together AI, Groq, OpenRouter, Azure OpenAI, or any endpoint that implements the `/v1/chat/completions` spec.

No modifications to KiroCrew source code required. Works via runtime monkey-patching of the provider registry.

## How It Works

```
KiroCrew Dashboard
      ↓
KiroCrew Gateway  (sessions, memory, cron, approval ladder)
      ↓
OpenAIProvider  ← replaces AcpProvider at runtime
      ↓
OpenAI-compatible HTTP API  (Ollama / vLLM / OpenAI / Azure / ...)

Embedding path (separate):
Knowledge / VectorMemory → OpenAIEmbeddingBackend → POST /v1/embeddings
```

The provider is installed by monkey-patching `ProviderRegistry.create_factory` and `LLMPool._create_worker` before the gateway starts. All chat sessions, subagents, cron jobs, and knowledge extraction routes are covered.

## Quick Start

### Prerequisites

- Python 3.10+
- [KiroCrew](https://github.com/kirocrew/kirocrew) installed (`pip install kirocrew`)
- An OpenAI-compatible API endpoint

> **⚠️ Important — same virtual environment required**
>
> `gateway.py` **must** be run in the same Python virtual environment where
> `kirocrew` is installed. If you get `ImportError: No module named 'kiro_crew'`,
> you are running in a different env.
>
> Verify your active env:
> ```bash
> pip show kirocrew
> # If this returns nothing, kirocrew is NOT in your current env.
>
> # Activate the correct venv:
> source /path/to/kirocrew-venv/bin/activate
> # Or use the venv's python directly:
> /path/to/kirocrew-venv/bin/python gateway.py
> ```

### Install Dependencies

```bash
pip install httpx ddgs
```

### Run the Gateway

```bash
# Ollama (local)
OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_API_KEY=ollama \
OPENAI_MODEL=qwen2.5-coder:32b \
python gateway.py

# OpenAI
OPENAI_API_KEY=sk-... \
OPENAI_MODEL=gpt-4o \
python gateway.py

# vLLM
OPENAI_BASE_URL=http://localhost:8000/v1 \
OPENAI_MODEL=mistral-7b-instruct \
python gateway.py

# LiteLLM proxy (multi-provider)
OPENAI_BASE_URL=http://localhost:4000/v1 \
OPENAI_API_KEY=anything \
OPENAI_MODEL=anthropic/claude-opus-4 \
python gateway.py
```

The dashboard works identically — memory, cron, subagents, knowledge library, all functional.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint URL |
| `OPENAI_API_KEY` | *(required)* | API key for authentication |
| `OPENAI_MODEL` | `gpt-4o` | Default model for chat sessions |
| `OPENAI_MAX_TOKENS` | `8192` | Max output tokens per turn |
| `OPENAI_CONTEXT_WINDOW` | *(auto-resolved)* | Model context window size. Auto-detected from `/v1/models` endpoint, with a built-in fallback map for common models. Set explicitly if auto-detection fails. |
| `OPENAI_SYSTEM_PROMPT` | *(empty)* | Override the default system prompt |
| `OPENAI_KNOWLEDGE_MODEL` | *(same as `OPENAI_MODEL`)* | Separate model for knowledge extraction (entity/relation extraction). Useful when you want a cheaper/faster model for background knowledge processing. |
| `OPENAI_COMPACTION_THRESHOLD` | `80` | Context usage percentage that triggers auto-compaction (0 to disable) |
| `OPENAI_COMPACTION_KEEP_RECENT` | `6` | Number of recent messages to preserve during compaction |
| `EMBEDDING_BASE_URL` | *(same as `OPENAI_BASE_URL`)* | Embedding API endpoint URL |
| `EMBEDDING_API_KEY` | *(same as `OPENAI_API_KEY`)* | Auth key for embedding API |
| `EMBEDDING_MODEL` | `nvidia/nvidia/nemotron-3-embed-1b` | Embedding model id |
| `EMBEDDING_DIM` | `2048` | Vector dimensionality (must match stored vectors) |

## Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| Text streaming | ✅ | SSE → `EVENT_TEXT_CHUNK` |
| Tool / function calling | ✅ | OpenAI function-calling spec |
| Thinking / reasoning | ✅ | `reasoning_content` field (o1, DeepSeek-R1, MiMo) |
| KiroCrew approval ladder | ✅ | `permission_request` → approve/reject |
| Tool cancel | ✅ | `asyncio.Event` |
| Context usage tracking | ✅ | `prompt_tokens / context_window` |
| KiroCrew memory & cron | ✅ | Unaffected (KiroCrew layer) |
| KiroCrew subagents | ✅ | Each subagent gets its own `OpenAIProvider` |
| Knowledge extraction | ✅ | `OpenAIWorker` — non-streaming HTTP, no subprocess |
| Context compaction | ✅ | Auto-summarizes old messages when context fills up |
| Remote embedding backend | ✅ | Replaces bundled llama.cpp — saves ~610 MB RAM + 0% CPU |
| Background session routing | ✅ | Auto-title/link-summary use OpenAI (Patch 5) |
| Web search | ✅ | DuckDuckGo via `ddgs` — titles, URLs, snippets |
| MCP tool routing | ✅ | Registry-based dispatch: 13 local handlers + 115 dynamic MCP tools |

## Architecture

### Provider Path (Chat Sessions)

The main chat path goes through KiroCrew's `SessionManager` → `ProviderRegistry`. The `install()` function replaces the factory so every session gets an `OpenAIProvider` instead of the default `AcpProvider`.

```
SessionManager
  → ProviderRegistry.create_factory()  ← patched
    → OpenAIProvider(base_url, api_key, model)
      → httpx POST /chat/completions (streaming)
```

### Knowledge Extraction Path

Knowledge extraction uses a separate `LLMPool` with long-lived workers. The `install()` function also patches `LLMPool._create_worker` to return `OpenAIWorker` instances (pure HTTP, no subprocess) instead of `AcpClient`-backed workers.

```
LLMPool._create_worker()  ← monkey-patched
  → OpenAIWorker()
    → httpx POST /chat/completions (non-streaming)
```

This is lighter than the original ACP workers — no kiro-cli subprocess spawned, just HTTP calls.

### Embedding Path (Knowledge + Vector Memory)

By default, KiroCrew bundles a llama.cpp runtime (~610 MB) that loads a local embedding model (Qwen3-Embedding-0.6B) and runs inference on CPU. This provider replaces it with a remote HTTP backend:

```
Knowledge / VectorMemory
  → InProcessEmbedder
    → OpenAIEmbeddingBackend  ← replaces LlamaCppEmbedder
      → POST /v1/embeddings (remote API)
```

This eliminates:
- **~610 MB RAM** (no model weights loaded)
- **~189% CPU** (no local matrix multiplication on 2-core ARM)

Configure via `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`. If the remote endpoint is unreachable, embedding returns `None` and KiroCrew falls back to keyword/FTS search.

### Background Session Path

KiroCrew runs lightweight background LLM calls for auto-title generation and nav bar link summary resolution. Without Patch 5, those calls route through `SessionManager._bg_provider_is_kiro()` → `AcpSessionHandle` → kiro-cli → Anthropic API, **bypassing** all openai_provider patches. This caused `AcpError: The monthly usage limit has been reached` errors (100+ per day) when the Anthropic quota was exhausted.

```
run_bg_oneliner()
  → sessions.get_bg_session()
    → SessionManager._bg_provider_is_kiro()  ← patched to return False
      → _ensure_background()
        → OpenAIProvider (via patched factory)
          → 9router → free model endpoint
```

### MCP Tool Routing

The OpenAI provider exposes all KiroCrew MCP tools to the model via a two-layer dispatch system:

```
OpenAI Model
  → McpToolExecutor.execute(name, args)
    ├── 1. Local Registry (13 handlers, O(1), zero-latency)
    │     read, write, edit, grep, glob, find, list_dir,
    │     execute_bash, web_fetch, web_search,
    │     ask_question, spawn_run, spawn_list
    │
    └── 2. MCP Stdio Protocol (115 tools from managed servers)
          kirocrew-core (65): learn_add, cron_*, task_run, wait, ...
          kirocrew-cron  (8): cron_add, cron_list, cron_update, ...
          kirocrew-computer:  disabled (macOS only)
```

**Discovery** happens at startup without spawning subprocesses: managed server modules (`kiro_crew.mcp_core`, `kiro_crew.mcp_cron`) are imported in-process and their `_list_tools()` functions return full tool schemas. Non-managed servers (e.g. `playwright-mcp`) are discovered via `mcp_discovery.list_servers()` with MCP protocol probing.

**Execution** has three tiers:
1. **Local registry** — Python reimplementation of kiro-cli's built-in tools, plus gateway-proxied handlers for `ask_question` (session_directive) and `spawn_run`/`spawn_list` (HTTP to gateway API)
2. **MCP stdio** — spawns the target server as a subprocess, sends `tools/call` via JSON-RPC, parses result
3. **Error with hint** — lists all available tools when a tool is not found

Adding new local handlers requires only subclassing `McpHandler` and calling `McpHandlerRegistry.register()` — no existing code needs modification (Open/Closed Principle).

### Code Review Sage

The code review sage (`kirocrew-crsage`) uses `AcpRuntime` directly (not via `SessionManager`) and requires tool execution capabilities (shell, `gh` CLI). It is **not** patched by this provider and continues to use kiro-cli with the default Claude backend.

## Configuration

### Switching Models

Set `OPENAI_MODEL` to any model your endpoint supports:

```bash
OPENAI_MODEL=deepseek-r1        # DeepSeek R1
OPENAI_MODEL=gpt-4o             # OpenAI GPT-4o
OPENAI_MODEL=llama3.1:70b       # Ollama local
OPENAI_MODEL=my-custom-model    # Any name your proxy recognizes
```

### Separate Knowledge Model

For cost optimization, use a cheaper model for knowledge extraction:

```bash
OPENAI_MODEL=gpt-4o                    # Main chat: full-featured
OPENAI_KNOWLEDGE_MODEL=gpt-4o-mini     # Knowledge extraction: cheaper
```

### Context Window

The provider auto-detects the context window from your API's `/v1/models` endpoint. If that fails, it falls back to a built-in map of known models. Override manually:

```bash
OPENAI_CONTEXT_WINDOW=128000
```

### Custom System Prompt

```bash
OPENAI_SYSTEM_PROMPT="You are a helpful coding assistant specialized in Python."
```

### Remote Embedding Backend

Replace the bundled llama.cpp model with a remote embedding API:

```bash
EMBEDDING_BASE_URL=http://your-api:20128/v1
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL=nvidia/nvidia/nemotron-3-embed-1b
EMBEDDING_DIM=2048
```

Any OpenAI-compatible `/v1/embeddings` endpoint works. The backend validates output dimensions and retries transient failures (3 attempts with exponential backoff).

> **Note**: Changing the embedding model invalidates stored vectors. KiroCrew will re-embed them in the background via the new endpoint.

## File Structure

```
KiroCrew-OpenAI-Compatible/
├── README.md
├── LICENSE
├── CONTRIBUTING.md              # Contribution guidelines
├── SECURITY.md                  # Security policy and reporting
├── CODE_OF_CONDUCT.md           # Community code of conduct
├── .gitignore
├── .env.example                 # Template for environment variables
├── .github/
│   ├── PULL_REQUEST_TEMPLATE.md # PR template
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md        # Bug report template
│       └── feature_request.md   # Feature request template
├── gateway.py                   # Entry point — start KiroCrew with OpenAI provider
└── openai_provider/             # Python package
    ├── __init__.py              # Public API: install()
    ├── _config.py               # Shared config, env vars, known model map
    ├── provider.py              # OpenAIProvider (LLMProvider implementation)
    ├── install.py               # Monkey-patches KiroCrew (5 patches: factory, config, pool, embed, bg)
    ├── embedding_backend.py     # OpenAIEmbeddingBackend (replaces bundled llama.cpp)
    ├── mcp_executor.py          # McpToolExecutor — bridges tool calls → MCP dispatch
    ├── mcp_handler_registry.py  # Registry-based MCP handler system (13 local handlers)
    ├── openai_worker.py         # OpenAIWorker for knowledge extraction
    ├── test_openai_worker.py    # Tests for OpenAIWorker + shared config
    └── test_tools.py            # Tests for MCP tool execution
```

## Custom ToolExecutor

You can supply your own tool executor to handle tool calls:

```python
from openai_provider import install
from openai_provider.provider import ToolExecutor

class MyExecutor(ToolExecutor):
    def tool_definitions(self):
        return [{"type": "function", "function": {"name": "my_tool", ...}}]

    async def execute(self, name, args):
        if name == "my_tool":
            return "result"

install(tool_executor=MyExecutor())
```

## Known Limitations

1. **Session persistence**: OpenAIProvider keeps conversation history in-memory. On gateway restart, history is lost (no kiro-cli session file). A future version may serialize `_messages` to disk via `session_key`.

2. **Multimodal**: Image attachments are not passed through to vision-capable APIs yet.

3. **Code review sage**: Uses `AcpRuntime` directly (requires tool execution via kiro-cli). Not covered by this provider — continues to use the default Claude backend.

4. **Web search dependency**: The `web_search` handler requires the `ddgs` package (`pip install ddgs`). Falls back to `duckduckgo_search` if unavailable.

5. **MCP stdio overhead**: Non-local tools (cron_add, learn_add, etc.) spawn a subprocess per call (~200-500ms). Acceptable for infrequent calls; local handlers have zero overhead.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and the PR workflow.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Run the tests: `python -m unittest openai_provider.test_openai_worker openai_provider.test_tools -v`
4. Submit a pull request

## License

MIT
