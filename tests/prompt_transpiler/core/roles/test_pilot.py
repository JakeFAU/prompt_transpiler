from unittest.mock import AsyncMock, patch

import pytest

from prompt_transpiler.core.roles.pilot import DefaultPilot
from prompt_transpiler.dto.models import (
    LLMResponse,
    Message,
    Model,
    ModelProviderType,
    PromptPayload,
    PromptStyle,
    Provider,
    TokenUsage,
)
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt


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


@pytest.mark.asyncio
async def test_pilot_success(mock_model):
    with patch("prompt_transpiler.core.roles.pilot.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content="Candidate Response", model_name="gpt-4", usage=TokenUsage(total_tokens=100)
        )
        mock_get_provider.return_value = mock_provider

        pilot = DefaultPilot()
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="Prompt")]),
            model=mock_model,
        )

        result = await pilot.test_candidate(candidate)

        assert result.response == "Candidate Response"
        mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_pilot_failure(mock_model):
    with patch("prompt_transpiler.core.roles.pilot.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("LLM Error")
        mock_get_provider.return_value = mock_provider

        pilot = DefaultPilot()
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="Prompt")]),
            model=mock_model,
        )

        result = await pilot.test_candidate(candidate)

        assert result is not None
        assert result.response is not None
        assert "ERROR: Execution failed" in result.response
