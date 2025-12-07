from unittest.mock import AsyncMock, patch

import pytest

from prompt_compiler.core.exceptions import ArchitectureError
from prompt_compiler.core.roles.architect import GPTArchitect
from prompt_compiler.dto.models import (
    IntermediateRepresentation,
    IntermediateRepresentationData,
    IntermediateRepresentationMeta,
    IntermediateRepresentationSpec,
    Model,
    ModelProviderType,
    PromptStyle,
    Provider,
)


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
    with patch("prompt_compiler.core.roles.architect.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = "Optimized Prompt"
        mock_get_provider.return_value = mock_provider

        architect = GPTArchitect()
        candidate = await architect.design_prompt(mock_ir, mock_model)

        assert candidate.prompt == "Optimized Prompt"
        assert candidate.model == mock_model
        mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_architect_design_prompt_with_feedback(mock_ir, mock_model):
    with patch("prompt_compiler.core.roles.architect.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = "Optimized Prompt"
        mock_get_provider.return_value = mock_provider

        architect = GPTArchitect()
        await architect.design_prompt(mock_ir, mock_model, feedback="Fix this")

        call_args = mock_provider.generate.call_args
        assert "Fix this" in call_args.kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_architect_failure(mock_ir, mock_model):
    with patch("prompt_compiler.core.roles.architect.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("LLM Error")
        mock_get_provider.return_value = mock_provider

        architect = GPTArchitect()
        with pytest.raises(ArchitectureError, match="Architect failed to generate prompt"):
            await architect.design_prompt(mock_ir, mock_model)
