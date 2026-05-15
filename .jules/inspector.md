## 2026-05-15 - LLMAdjudicator evaluate valid JSON but not dict
**Edge Case:** LLMAdjudicator receives valid JSON that is not a dictionary (e.g. a list or boolean).
**Learning:** In `src/prompt_transpiler/core/scoring.py`, `LLMAdjudicator.evaluate` parses JSON safely, but `_apply_pairwise_verdicts` and `_apply_numeric_scores` expect a dictionary. If it's a list or boolean, an inner `AttributeError` or `TypeError` occurs, which the outer `try...except Exception as e:` block catches and returns 0.0 safely.
**Action:** Mock `provider.generate` to return valid JSON that does not match the expected schema (like a list `[1, 2]` or string `"not a dict"`) to ensure it fails gracefully to 0.0.
