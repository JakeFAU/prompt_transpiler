## 2024-05-18 - [Hugging Face network failure handling]
**Edge Case:** The `HuggingFaceAdapter.available_models()` in `src/prompt_transpiler/llm/huggingface.py` catches a generic `Exception` when `list_models` fails (e.g. network timeout or API error) and returns an empty list `[]`. But there's no test covering this edge case.
**Learning:** Returning `[]` safely mitigates an application crash when Hugging Face hub is down or offline. The system correctly fails safely without bubbling up the exception.
**Action:** Need to write a test where `list_models` fails with a mock Exception to verify it returns `[]` safely.
