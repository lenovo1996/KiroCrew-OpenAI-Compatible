# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest `master` | ✅ |
| Older commits | ❌ |

This project is a runtime patch layer, not a standalone application. Security fixes land on `master` and users are expected to pull the latest version.

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Instead, email the maintainer directly or use [GitHub's private vulnerability reporting](https://github.com/lenovo1996/KiroCrew-OpenAI-Compatible/security/advisories/new).

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

### Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 5 business days
- **Fix or mitigation**: depends on severity, typically within 1-2 weeks

## Security Considerations

### API Key Handling

- API keys are passed via environment variables (`OPENAI_API_KEY`, `EMBEDDING_API_KEY`), never stored in code or config files
- The gateway does not log API keys
- Users must ensure their deployment environment protects these variables

### Monkey-Patching

This project replaces KiroCrew's provider at runtime via monkey-patching. This means:

- All LLM traffic (chat, knowledge extraction, embeddings) flows through this code
- A compromised provider could intercept or modify prompts and responses
- Always run from a trusted checkout — verify the integrity of your local copy

### MCP Tool Execution

The MCP executor can run shell commands (`execute_bash`) and read/write files. When exposed to untrusted input:

- KiroCrew's built-in approval ladder provides a security gate for destructive operations
- The gateway binds to `localhost` by default — do not expose it publicly without additional auth

### Dependencies

- `httpx` — HTTP client for API calls
- `ddgs` — DuckDuckGo web search (optional, for `web_search` handler)

Audit dependencies regularly. Run `pip audit` or check for known CVEs.

## Best Practices for Deployments

1. **Never expose the gateway to the public internet** without a reverse proxy with authentication
2. **Use HTTPS** for all API endpoints — set `OPENAI_BASE_URL` to an `https://` URL
3. **Rotate API keys** periodically
4. **Run in a dedicated virtualenv** to avoid dependency conflicts
5. **Monitor logs** for unexpected tool calls or API errors