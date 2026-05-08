## 2026-05-08 - Cache Async LLM Clients
**Learning:** Async LLM clients (like AsyncOpenAI) have significant initialization overhead. Creating them repeatedly in factory functions degrades performance.
**Action:** Use `@functools.cache` on globally accessed factory functions like `get_llm_provider` to reuse instances, and ensure test isolation by calling `.cache_clear()` in an `autouse=True` pytest fixture.
