from unittest.mock import AsyncMock, patch

import pytest

from prompt_transpiler.core.exceptions import ArchitectureError
from prompt_transpiler.core.roles.architect import GPTArchitect
from prompt_transpiler.dto.models import (
    IntermediateRepresentation,
    IntermediateRepresentationData,
    IntermediateRepresentationMeta,
    IntermediateRepresentationSpec,
    LLMResponse,
    Model,
    ModelProviderType,
    PromptPayload,
    PromptStyle,
    Provider,
    TokenUsage,
)


@pytest.fixture
def mock_model():
    return Model(
        provider=Provider(provider="openai", provider_type=ModelProviderType.API),
        model_name="gpt-4",
        supports_system_messages=True,
        supports_system_instructions=True,
        supports_structured_outputs=False,
        context_window_size=8192,
        prompt_style=PromptStyle.MARKDOWN,
        supports_json_mode=True,
        prompting_tips="tips",
    )


@pytest.fixture
def mock_ir(mock_model):
    return IntermediateRepresentation(
        meta=IntermediateRepresentationMeta(source_model=mock_model, target_model=mock_model),
        spec=IntermediateRepresentationSpec(
            primary_intent="intent",
            tone_voice="tone",
            domain_context="domain",
            constraints=["c1"],
            input_format="text",
            output_schema="text",
        ),
        data=IntermediateRepresentationData(few_shot_examples=[{"input": "i", "output": "o"}]),
    )


@pytest.mark.asyncio
async def test_architect_design_prompt_success(mock_ir, mock_model):
    with patch("prompt_transpiler.core.roles.architect.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content="Optimized Prompt", model_name="gpt-4-turbo", usage=TokenUsage(total_tokens=100)
        )
        mock_get_provider.return_value = mock_provider

        architect = GPTArchitect()
        candidate = await architect.design_prompt(mock_ir, mock_model)

        assert "Optimized Prompt" in candidate.prompt
        assert candidate.model == mock_model
        assert isinstance(candidate.payload, PromptPayload)
        mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_architect_design_prompt_routing_system(mock_ir, mock_model):
    mock_model.supports_system_instructions = True

    with patch("prompt_transpiler.core.roles.architect.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content="Optimized Prompt", model_name="gpt-4-turbo", usage=TokenUsage(total_tokens=100)
        )
        mock_get_provider.return_value = mock_provider

        architect = GPTArchitect()
        candidate = await architect.design_prompt(mock_ir, mock_model)

        messages = candidate.payload.messages
        assert any(m.role == "system" for m in messages)
        assert any(m.role == "user" for m in messages)

        system_msg = next(m for m in messages if m.role == "system")
        user_msg = next(m for m in messages if m.role == "user")

        assert "intent" in system_msg.content
        assert "tone" in system_msg.content
        assert "c1" in system_msg.content

        assert "domain" in user_msg.content
        assert "Optimized Prompt" in user_msg.content


@pytest.mark.asyncio
async def test_architect_design_prompt_routing_flat(mock_ir, mock_model):
    mock_model.supports_system_instructions = False

    with patch("prompt_transpiler.core.roles.architect.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content="Optimized Prompt", model_name="gpt-4-turbo", usage=TokenUsage(total_tokens=100)
        )
        mock_get_provider.return_value = mock_provider

        architect = GPTArchitect()
        candidate = await architect.design_prompt(mock_ir, mock_model)

        messages = candidate.payload.messages
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "intent" in messages[0].content
        assert "Optimized Prompt" in messages[0].content


@pytest.mark.asyncio
async def test_architect_design_prompt_structured_output(mock_ir, mock_model):
    mock_model.supports_structured_outputs = True
    mock_ir.spec.output_schema = "JSON schema for user"

    with patch("prompt_transpiler.core.roles.architect.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content="Optimized Prompt", model_name="gpt-4-turbo", usage=TokenUsage(total_tokens=100)
        )
        mock_get_provider.return_value = mock_provider

        architect = GPTArchitect()
        candidate = await architect.design_prompt(mock_ir, mock_model)

        assert candidate.payload.response_format is not None
        assert candidate.payload.response_format["type"] == "json_schema"
        assert "JSON schema for user" in candidate.payload.response_format["json_schema"]["schema"]


@pytest.mark.asyncio
async def test_architect_design_prompt_with_feedback(mock_ir, mock_model):
    with patch("prompt_transpiler.core.roles.architect.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content="Optimized Prompt", model_name="gpt-4-turbo", usage=TokenUsage(total_tokens=100)
        )
        mock_get_provider.return_value = mock_provider

        architect = GPTArchitect()
        await architect.design_prompt(mock_ir, mock_model, feedback="Fix this")

        call_args = mock_provider.generate.call_args
        assert "Fix this" in call_args.kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_architect_failure(mock_ir, mock_model):
    with patch("prompt_transpiler.core.roles.architect.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("LLM Error")
        mock_get_provider.return_value = mock_provider

        architect = GPTArchitect()
        with pytest.raises(ArchitectureError, match="Architect failed to generate prompt"):
            await architect.design_prompt(mock_ir, mock_model)
