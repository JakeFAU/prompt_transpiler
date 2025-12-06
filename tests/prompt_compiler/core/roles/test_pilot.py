from unittest.mock import AsyncMock, patch

import pytest

from prompt_complier.core.roles.pilot import DefaultPilot
from prompt_complier.dto.models import Model, ModelProviderType, PromptStyle, Provider
from prompt_complier.llm.prompts.prompt_objects import CandidatePrompt


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
    with patch("prompt_complier.core.roles.pilot.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = "Candidate Response"
        mock_get_provider.return_value = mock_provider

        pilot = DefaultPilot()
        candidate = CandidatePrompt(prompt="Prompt", model=mock_model)

        result = await pilot.test_candidate(candidate)

        assert result.response == "Candidate Response"
        mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_pilot_failure(mock_model):
    with patch("prompt_complier.core.roles.pilot.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("LLM Error")
        mock_get_provider.return_value = mock_provider

        pilot = DefaultPilot()
        candidate = CandidatePrompt(prompt="Prompt", model=mock_model)

        result = await pilot.test_candidate(candidate)

        assert result is not None
        assert result.response is not None
        assert "ERROR: Execution failed" in result.response
