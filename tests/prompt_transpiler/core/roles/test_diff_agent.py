import json
from unittest.mock import AsyncMock, patch

import pytest

from prompt_transpiler.core.roles.diff import SemanticDiffAgent
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
        model_name="gpt-4",
        supports_system_messages=True,
        context_window_size=8192,
        prompt_style=PromptStyle.MARKDOWN,
        supports_json_mode=True,
        prompting_tips="tips",
    )


@pytest.mark.asyncio
async def test_diff_agent_success(mock_model):
    with patch("prompt_transpiler.core.roles.diff.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(
                {
                    "summary": "Condensed the request and tightened format.",
                    "key_differences": [
                        "Tone shifted to concise",
                        "Added explicit JSON structure",
                    ],
                    "rationale": "Helps the target model respond consistently.",
                }
            ),
            model_name="gpt-4o-mini",
            usage=TokenUsage(total_tokens=50),
        )
        mock_get_provider.return_value = mock_provider

        agent = SemanticDiffAgent()
        original = OriginalPrompt(
            payload=PromptPayload(messages=[Message(role="user", content="Start")]),
            model=mock_model,
        )
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="End")]), model=mock_model
        )

        summary = await agent.summarize_diff(original, candidate)

        assert "Condensed" in summary
        assert "Key differences" in summary
        assert candidate.diff_summary == summary
        mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_diff_agent_invalid_json(mock_model):
    with patch("prompt_transpiler.core.roles.diff.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content="Plain text diff", model_name="gpt-4o-mini", usage=TokenUsage(total_tokens=10)
        )
        mock_get_provider.return_value = mock_provider

        agent = SemanticDiffAgent()
        original = OriginalPrompt(
            payload=PromptPayload(messages=[Message(role="user", content="Start")]),
            model=mock_model,
        )
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="End")]), model=mock_model
        )

        summary = await agent.summarize_diff(original, candidate)

        assert summary == "Plain text diff"
        assert candidate.diff_summary == "Plain text diff"


@pytest.mark.asyncio
async def test_diff_agent_failure(mock_model):
    with patch("prompt_transpiler.core.roles.diff.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("LLM Error")
        mock_get_provider.return_value = mock_provider

        agent = SemanticDiffAgent()
        original = OriginalPrompt(
            payload=PromptPayload(messages=[Message(role="user", content="Start")]),
            model=mock_model,
        )
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="End")]), model=mock_model
        )

        summary = await agent.summarize_diff(original, candidate)

        assert summary == ""
        assert candidate.diff_summary is None
        mock_provider.generate.assert_called_once()
