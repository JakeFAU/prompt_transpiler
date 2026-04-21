## 2025-04-21 - Unhandled None in Model Resolution
**Edge Case:** Transpile job request payload unexpectedly provides `None` for `source_model` or `target_model`.
**Learning:** The `ModelRegistry.get_model` fallback logic attempts to call `.lower()` on the provided model name string. If `None` is provided, it causes an unhandled `AttributeError` (e.g., `'NoneType' object has no attribute 'lower'`), causing the entire compile job to fail with a stack trace instead of a graceful schema validation error or graceful failure.
**Action:** Always test schema validation layers and the core boundaries processing string identifiers to ensure `None` or empty dictionary structures are safely rejected or defaulted before calling string methods.
