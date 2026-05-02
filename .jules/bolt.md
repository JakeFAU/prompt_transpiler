## 2026-05-02 - Adding lru_cache to get_llm_provider
**Learning:** Applying lru_cache to global factory functions (get_llm_provider) reduces unnecessary initialization overhead in hot paths like Pilot and Judge.
**Action:** When memoizing global singletons in tests, add an autouse=True fixture in conftest.py that calls cache_clear() on the factory. This ensures clean state per-test.
