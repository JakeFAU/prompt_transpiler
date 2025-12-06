import json
from unittest.mock import AsyncMock, patch

import pytest

from prompt_complier.core.exceptions import DecompilationError
from prompt_complier.core.roles.decompiler import GeminiDecompiler
from prompt_complier.dto.models import Model, ModelProviderType, PromptStyle, Provider
from prompt_complier.llm.prompts.prompt_objects import OriginalPrompt


@pytest.fixture
def mock_model():
    return Model(
        provider=Provider(provider="openai", provider_type=ModelProviderType.API),
        model_name="gpt-4",
        supports_system_messages=True,
        context_window_size=8192,
        prompt_style=PromptStyle.MARKDOWN,
        supports_json_mode=True,
        prompting_tips="tips",
    )


@pytest.fixture
def mock_original_prompt(mock_model):
    return OriginalPrompt(prompt="Original Prompt", model=mock_model)


@pytest.mark.asyncio
async def test_decompiler_success(mock_original_prompt, mock_model):
    with patch("prompt_complier.core.roles.decompiler.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        response_data = {
            "primary_intent": "intent",
            "tone_voice": "tone",
            "domain_context": "domain",
            "constraints": ["c1"],
            "input_format": "text",
            "output_schema": "text",
            "few_shot_examples": [{"input": "i", "output": "o"}],
        }
        mock_provider.generate.return_value = json.dumps(response_data)
        mock_get_provider.return_value = mock_provider

        decompiler = GeminiDecompiler()
        ir = await decompiler.decompile(mock_original_prompt, mock_model)

        assert ir.spec.primary_intent == "intent"
        assert len(ir.data.few_shot_examples) == 1
        mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_decompiler_invalid_json(mock_original_prompt, mock_model):
    with patch("prompt_complier.core.roles.decompiler.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = "Not JSON"
        mock_get_provider.return_value = mock_provider

        decompiler = GeminiDecompiler()
        with pytest.raises(DecompilationError, match="Decompiler output was not valid JSON"):
            await decompiler.decompile(mock_original_prompt, mock_model)


@pytest.mark.asyncio
async def test_decompiler_failure(mock_original_prompt, mock_model):
    with patch("prompt_complier.core.roles.decompiler.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("LLM Error")
        mock_get_provider.return_value = mock_provider

        decompiler = GeminiDecompiler()
        with pytest.raises(DecompilationError, match="Decompiler failed during generation"):
            await decompiler.decompile(mock_original_prompt, mock_model)
