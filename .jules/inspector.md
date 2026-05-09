## 2026-05-09 - LLMAdjudicator JSON Parsing Edge Case
**Edge Case:** The LLM returning valid JSON that is an array or boolean instead of a dictionary to `LLMAdjudicator.evaluate()`.
**Learning:** The system parses valid JSON but fails with an AttributeError/TypeError when treating it as a dictionary, but safely catches it in a broad `except Exception` block, returning 0.0.
**Action:** Test for unexpected valid JSON structures (arrays, booleans, nulls) when parsing LLM outputs across the project.
