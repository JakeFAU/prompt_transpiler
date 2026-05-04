## 2025-05-04 - Caching LLM Provider Factory
**Learning:** The get_llm_provider factory function creates a new LLM provider instance on every call. This causes unnecessary overhead, especially since the provider objects are stateless. Adding @lru_cache significantly reduces object allocations.
**Action:** Use @lru_cache for stateless factory functions to prevent redundant object instantiation.
