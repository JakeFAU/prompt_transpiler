## 2025-03-05 - LLMAdjudicator Non-Dict JSON Edge Case
**Edge Case:** The LLM returns a valid JSON string that parses into a list or boolean instead of the expected dictionary payload.
**Learning:** The outer `try...except Exception` catches `AttributeError` caused by calling `data.get()` on lists/booleans inside `_apply_numeric_scores`, returning `0.0` safely.
**Action:** Add edge-case unit tests to explicitly cover LLM JSON responses that parse correctly but are not dictionaries (e.g., lists, booleans, integers).
