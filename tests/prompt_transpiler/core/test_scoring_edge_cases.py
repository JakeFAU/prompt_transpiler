from unittest.mock import AsyncMock, patch

import pytest

from prompt_transpiler.core.scoring import LLMAdjudicator
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
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt, OriginalPrompt


@pytest.fixture
def mock_model():
    return Model(
        provider=Provider(provider="openai", provider_type=ModelProviderType.API),
        model_name="gpt-4o",
        supports_system_messages=True,
        context_window_size=8000,
        prompt_style=PromptStyle.MARKDOWN,
        supports_json_mode=False,
        prompting_tips="",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "json_payload", ['["not", "a", "dict"]', "true", "42", "null", '"a string"']
)
async def test_llm_adjudicator_malformed_json_types(mock_model, json_payload):
    """
    Test that LLMAdjudicator gracefully handles a response that is valid JSON
    but the wrong type (e.g. list, bool, int instead of dict).
    """
    with patch("prompt_transpiler.core.scoring.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content=json_payload, model_name="gpt-4o", usage=TokenUsage(total_tokens=100)
        )
        mock_get_provider.return_value = mock_provider

        judge = LLMAdjudicator()
        original = OriginalPrompt(
            payload=PromptPayload(messages=[Message(role="user", content="orig")]), model=mock_model
        )
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="cand")]), model=mock_model
        )

        score = await judge.evaluate(candidate, original)
        assert score == 0.0
