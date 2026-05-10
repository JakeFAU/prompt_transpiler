## 2026-05-10 - LLM JSON Response Type Weakness
**Edge Case:** The LLM returns valid JSON that parses to a list, bool, or int instead of a dictionary.
**Learning:** The system uses `json.loads` directly without schema validation (like Marshmallow) and assumes the result is a dict. This causes a `TypeError` or `AttributeError` which is caught by a generic `Exception` block, returning a 0.0 score.
**Action:** Test other LLM integration points or external API consumers for missing schema validation when parsing JSON.
