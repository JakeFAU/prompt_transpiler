
## 2026-05-02 - LLMAdjudicator Malformed JSON Types
**Edge Case:** The LLM returns a valid JSON string, but the parsed dictionary contains unexpected data types (e.g., lists, booleans, or dicts) for expected string/list fields.
**Learning:** The LLMAdjudicator gracefully degrades these malformed types to None and assigns default fallback scores like TIE_SCORE without raising exceptions.
**Action:** Always test that components parsing LLM JSON responses validate data types and gracefully handle or reject unexpected structures, rather than assuming standard schemas.
