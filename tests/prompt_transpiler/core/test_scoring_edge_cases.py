import pytest

from prompt_transpiler.core.scoring import LLMAdjudicator
from prompt_transpiler.dto.models import (
    Message,
    Model,
    ModelProviderType,
    PromptPayload,
    PromptStyle,
    Provider,
)
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt, OriginalPrompt


@pytest.mark.asyncio
async def test_llmadjudicator_evaluate_json_list(mocker):
    """
    Edge Case: The LLM returns a valid JSON list ("[]") instead of a dictionary.
    This tests that LLMAdjudicator safely handles JSON payloads that are not objects.
    """
    mock_provider = mocker.AsyncMock()
    mock_response = mocker.Mock()
    mock_response.content = "[]"
    mock_response.usage = {}
    mock_provider.generate.return_value = mock_response

    mocker.patch("prompt_transpiler.core.scoring.get_llm_provider", return_value=mock_provider)
    mocker.patch("prompt_transpiler.core.scoring.token_collector.add")

    adjudicator = LLMAdjudicator()
    payload = PromptPayload(messages=[Message(role="user", content="content")])
    provider = Provider(provider="openai", provider_type=ModelProviderType.API)
    model = Model(
        provider=provider,
        model_name="gpt-4",
        supports_system_messages=True,
        context_window_size=8000,
        prompt_style=PromptStyle.MARKDOWN,
        supports_json_mode=True,
        prompting_tips="",
    )
    candidate = CandidatePrompt(payload=payload, model=model, response="c_response")
    original = OriginalPrompt(payload=payload, model=model, response="o_response")

    result = await adjudicator.evaluate(candidate, original)
    assert result == 0.0


@pytest.mark.asyncio
async def test_llmadjudicator_evaluate_json_bool(mocker):
    """
    Edge Case: The LLM returns a valid JSON boolean ("true") instead of a dictionary.
    This tests that LLMAdjudicator safely handles JSON payloads that are not objects.
    """
    mock_provider = mocker.AsyncMock()
    mock_response = mocker.Mock()
    mock_response.content = "true"
    mock_response.usage = {}
    mock_provider.generate.return_value = mock_response

    mocker.patch("prompt_transpiler.core.scoring.get_llm_provider", return_value=mock_provider)
    mocker.patch("prompt_transpiler.core.scoring.token_collector.add")

    adjudicator = LLMAdjudicator()
    payload = PromptPayload(messages=[Message(role="user", content="content")])
    provider = Provider(provider="openai", provider_type=ModelProviderType.API)
    model = Model(
        provider=provider,
        model_name="gpt-4",
        supports_system_messages=True,
        context_window_size=8000,
        prompt_style=PromptStyle.MARKDOWN,
        supports_json_mode=True,
        prompting_tips="",
    )
    candidate = CandidatePrompt(payload=payload, model=model, response="c_response")
    original = OriginalPrompt(payload=payload, model=model, response="o_response")

    result = await adjudicator.evaluate(candidate, original)
    assert result == 0.0
