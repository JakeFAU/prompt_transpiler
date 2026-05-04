## 2025-02-20 - Sync docs for Pipeline and Judge exceptions
**Drift Found:** Critical pipeline entry points and evaluation methods were missing documentation for their return types and expected exceptions (e.g., `EvaluationError` and `RuntimeError`).
**Learning:** Developers often forget to document exceptions in the `Raises:` block when they are raised conditionally or from nested blocks.
**Action:** Always verify that `Raises:` and `Returns:` sections exist for public API entry points, especially when methods rely on side-effects (like modifying the Candidate object).
