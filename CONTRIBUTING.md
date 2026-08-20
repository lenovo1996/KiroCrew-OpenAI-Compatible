# Contributing to KiroCrew-OpenAI-Compatible

Thanks for your interest in contributing! This project lets you run [KiroCrew](https://github.com/kirocrew/kirocrew) with any OpenAI-compatible API.

## Getting Started

### Prerequisites

- Python 3.10+
- [KiroCrew](https://github.com/kirocrew/kirocrew) installed (`pip install kirocrew`)
- An OpenAI-compatible API endpoint (Ollama, vLLM, LiteLLM, etc.)

### Setup

```bash
# Clone the repo
git clone https://github.com/lenovo1996/KiroCrew-OpenAI-Compatible.git
cd KiroCrew-OpenAI-Compatible

# Install dependencies
pip install httpx ddgs

# Verify your KiroCrew env is active
pip show kirocrew
```

### Running Tests

```bash
# Unit tests
python -m unittest openai_provider.test_openai_worker openai_provider.test_tools -v

# Smoke test — start the gateway against a real endpoint
OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_API_KEY=ollama \
OPENAI_MODEL=qwen2.5-coder:32b \
python gateway.py
```

## How to Contribute

### Reporting Bugs

Open an [issue](https://github.com/lenovo1996/KiroCrew-OpenAI-Compatible/issues) with:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (Python version, KiroCrew version, API provider)

### Suggesting Features

Open an issue describing the use case. Explain what problem it solves and how it fits the "drop-in replacement" philosophy of this project.

### Submitting Changes

1. Fork the repository
2. Create a feature branch from `master`:
   ```bash
   git checkout -b feature/my-change
   ```
3. Make your changes
4. Run the tests and make sure they pass:
   ```bash
   python -m unittest openai_provider.test_openai_worker openai_provider.test_tools -v
   ```
5. Commit with a clear message:
   ```bash
   git commit -m "feat: add support for X"       # new feature
   git commit -m "fix: handle Y edge case"       # bug fix
   git commit -m "docs: update Z section"        # docs only
   ```
6. Push and open a pull request against `master`

### Code Style

- Follow existing patterns in the codebase
- Keep monkey-patches minimal and well-documented
- New MCP handlers: subclass `McpHandler` and register via `McpHandlerRegistry.register()`
- Environment variables: prefix with `OPENAI_` or `EMBEDDING_` and document in README

### Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes:

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding/updating tests |
| `refactor:` | Code restructuring (no behavior change) |
| `chore:` | Tooling, CI, dependencies |

## Architecture Notes

If you're modifying the provider's core, these are the key integration points:

1. **Provider patch** (`install.py`) — replaces `ProviderRegistry.create_factory` and `LLMPool._create_worker`
2. **MCP dispatch** (`mcp_handler_registry.py` + `mcp_executor.py`) — routes tool calls to local handlers or MCP stdio servers
3. **Embedding backend** (`embedding_backend.py`) — replaces bundled llama.cpp with remote HTTP

Changes to `install.py` or `provider.py` can break all KiroCrew sessions. Test carefully.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).