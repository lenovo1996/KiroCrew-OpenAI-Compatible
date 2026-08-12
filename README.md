# KiroCrew-OpenAI-Compatible

**Drop-in OpenAI-compatible backend for [KiroCrew](https://github.com/kirocrew/kirocrew).**

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
```

The provider is installed by monkey-patching `ProviderRegistry.create_factory` and `LLMPool._create_worker` before the gateway starts. All chat sessions, subagents, cron jobs, and knowledge extraction routes are covered.

## Quick Start

### Prerequisites

- Python 3.10+
- [KiroCrew](https://github.com/kirocrew/kirocrew) installed (`pip install kirocrew`)
- An OpenAI-compatible API endpoint

### Install Dependencies

```bash
pip install httpx
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
| MCP tool routing | ⚠️ | Best-effort via `McpToolExecutor` |

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

## File Structure

```
KiroCrew-OpenAI-Compatible/
├── README.md
├── LICENSE
├── .gitignore
├── gateway.py                   # Entry point — start KiroCrew with OpenAI provider
└── openai_provider/             # Python package
    ├── __init__.py              # Public API: install()
    ├── _config.py               # Shared config, env vars, known model map
    ├── provider.py              # OpenAIProvider (LLMProvider implementation)
    ├── install.py               # Monkey-patches KiroCrew ProviderRegistry + LLMPool
    ├── mcp_executor.py          # Bridges tool calls → KiroCrew MCP servers
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

2. **Context compaction**: Auto-compaction when context fills up is not yet implemented. The provider tracks context usage percentage but does not summarize old history automatically.

3. **Multimodal**: Image attachments are not passed through to vision-capable APIs yet.

4. **Code review sage**: Uses `AcpRuntime` directly (requires tool execution via kiro-cli). Not covered by this provider — continues to use the default Claude backend.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Run the tests: `python -m unittest test_openai_worker test_tools -v`
4. Submit a pull request

## License

MIT
