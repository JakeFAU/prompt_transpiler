## 2024-05-03 - LLMAdjudicator Malformed JSON Types Degradation
**Edge Case:** Valid JSON structure but with unexpected data types (lists/dicts instead of strings, booleans instead of numbers) returned from the LLM.
**Learning:** The LLMAdjudicator handles type mismatches gracefully by using `isinstance()` checks and degrading to default values like TIE_SCORE instead of throwing exceptions.
**Action:** When testing external LLM integrations, always include tests for structurally valid but type-mismatched JSON schemas.
