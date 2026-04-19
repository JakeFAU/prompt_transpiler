# Design Spec: Unified Payload & Capability-Aware Architecture

**Status:** Draft
**Date:** 2026-04-19
**Topic:** Support for `system_instructions`, `structured_outputs`, and dynamic model capabilities.

## 1. Executive Summary
This design evolves the Prompt Transpiler from a string-in/string-out tool into a payload-aware pipeline. It introduces a `PromptPayload` object to handle multiple message roles (system/user) and native `response_format` schemas. It also moves model capability definitions into an externalizable JSON registry.

## 2. Architecture Changes

### 2.1 Unified Prompt Payload
The core DTOs (`OriginalPrompt`, `CandidatePrompt`) will be updated to use a `PromptPayload` instead of a raw `prompt` string.

```python
@define
class Message:
    role: str  # "system", "user"
    content: str

@define
class PromptPayload:
    messages: list[Message]
    response_format: dict[str, Any] | None = None
```

### 2.2 Capability-Aware Model Registry
The `ModelRegistry` will load from `src/prompt_transpiler/core/models.json` by default.

**Key Model Attributes:**
- `supports_system_instructions`: (bool)
- `supports_structured_outputs`: (bool)
- `prompt_style`: (enum: markdown, xml, plain)
- `prompting_tips`: (string)

### 2.3 Multi-Stage Architect Pipeline
The `Architect` role will implement a two-stage process:
1. **Deconstruction:** Analyze the Agnostic IR (Intent, Constraints, Context).
2. **Synthesis & Routing:**
   - Route instructions to the `system` message if supported.
   - Route data/payload to the `user` message.
   - Map `output_schema` to `response_format` if supported, otherwise bake into text.

## 3. Data Flow

1. **Input (CLI/API):**
   - User provides string (wrapped in a `user` message automatically).
   - User provides `prompt.json` (parsed into `PromptPayload`).
2. **Decompiler:**
   - Consolidates all `PromptPayload` content into a single Agnostic IR.
3. **Architect:**
   - Generates a new `PromptPayload` optimized for the target model's specific capabilities.
4. **Output:**
   - CLI prints the `system` and `user` messages clearly.
   - If `--output-json` is used, returns the full payload object.

## 4. Model Registry (Initial models.json)
We will seed the registry with:
- **OpenAI:** GPT-4o, GPT-4o-mini (System + Structured Output support).
- **Gemini:** 2.0 Flash/Pro (System + JSON Schema support).
- **Anthropic:** Claude 3.5 Sonnet (System support, XML style).
- **HuggingFace:** Llama 3.3, Phi-4 (Agnostic fallbacks).

## 5. Success Criteria
- [ ] Successful transpilation of a JSON payload (system+user) to a single-string target.
- [ ] Successful transpilation of a single-string source to a multi-part target (system instructions + user data).
- [ ] `response_format` from source is preserved or optimized for target.
- [ ] Users can add a new model by editing a JSON file without recompiling.
