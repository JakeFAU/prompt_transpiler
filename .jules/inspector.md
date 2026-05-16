## 2024-05-16 - Edge Case in LLMAdjudicator parsing
**Edge Case:** The LLM returns valid JSON that parses into a list instead of a dict, causing an internal `TypeError` or `AttributeError` when dict methods are called.
**Learning:** The outer generic `Exception` catch block safely handles the internal failure and gracefully returns 0.0, avoiding a pipeline crash.
**Action:** When parsing LLM JSON responses, explicitly check the parsed type (e.g., `isinstance(data, dict)`) before attempting to access keys or methods.
