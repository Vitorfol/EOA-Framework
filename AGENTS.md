# AGENTS.md - EOA Project Guidelines

This file is automatically loaded by the AI assistant across the workspace.

## Core Directives for EOA Development
1. **Developer Experience (DX) First**:
   - Every API must feel clean, intuitive, and thoughtfully crafted: *"Whoever developed this really cared about making something clean and easy to use."*
   - Clear method names (examples: `explain`, `create_session`, `remove_agent`, `override_agent_prompt`).
   - Detailed docstrings with practical usage examples.
   - Comprehensive type hints for perfect IDE autocomplete.

2. **Critical Thinking & Honest Feedback**:
   - Do not assume: evaluate trade-offs critically and communicate openly.
   - Target conference: **ICSE** (International Conference on Software Engineering). High SE rigor, benchmarkability, and zero bloat.

3. **Clean, Modular Code (Core Ultra-Lightweight)**:
   - Core dependency must strictly be `pydantic >= 2.0`.
   - Providers (Gemini, OpenAI, Anthropic) and CLI tooling are optional extras.
   - Session store defaults to native `sqlite3` and `in-memory`.
   - CMA uses structured retrieval by default, with pluggable hooks for RAG if needed.
   - Decoupled fail-safe ("unplugged"): failures in LLM or validation degrade gracefully without crashing the client's host system.
