## 2024-05-24 - LLMAdjudicator Non-Dict JSON Handling
**Edge Case:** Mocked LLM response to return valid JSON that parses into a list, boolean, number, or string instead of a dictionary (e.g., `[]`, `true`).
**Learning:** `LLMAdjudicator` unconditionally uses the `in` operator and `.get()` method on parsed JSON. This raises `TypeError` (for booleans) or `AttributeError` (for lists).
**Action:** Always test parsers/evaluators with valid JSON that is not of the expected top-level type to ensure graceful error handling.
