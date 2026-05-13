
## 2026-05-13 - Cache LLM provider factory
**Learning:** Async LLM clients carry heavy initialization overhead and should be cached to prevent performance degradation when accessed frequently via factory methods.
**Action:** Apply @functools.cache to factory functions returning async clients, and ensure test isolation by calling cache_clear() in an autouse fixture.
