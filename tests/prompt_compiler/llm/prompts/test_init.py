import pytest

from prompt_complier.dto.models import Model, ModelProviderType, PromptStyle, Provider
from prompt_complier.llm.prompts import (
    OriginalPrompt,
    OriginalPromptSchema,
    TranspiledPrompt,
    TranspiledPromptSchema,
)


@pytest.fixture
def mock_model():
    provider = Provider(provider="OpenAI", provider_type=ModelProviderType.API, metadata={})
    return Model(
        provider=provider,
        model_name="gpt-4",
        supports_system_messages=True,
        context_window_size=8192,
        prompt_style=PromptStyle.MARKDOWN,
        supports_json_mode=True,
        prompting_tips="Be concise.",
        metadata={},
    )


def test_init_original_prompt_creation(mock_model):
    op = OriginalPrompt(
        prompt="Hello", model=mock_model, response_format={"type": "json"}, response="Hi"
    )
    assert op.prompt == "Hello"
    assert op.model == mock_model


def test_init_original_prompt_schema(mock_model):
    schema = OriginalPromptSchema()
    data = {
        "prompt": "Hello",
        "model": {
            "provider": {"provider": "OpenAI", "provider_type": "api", "metadata": {}},
            "model_name": "gpt-4",
            "supports_system_messages": True,
            "context_window_size": 8192,
            "prompt_style": "markdown",
            "supports_json_mode": True,
            "prompting_tips": "Be concise.",
            "metadata": {},
        },
        "response_format": {"type": "json"},
        "response": "Hi",
    }
    op = schema.load(data)
    assert isinstance(op, OriginalPrompt)
    assert op.prompt == "Hello"


def test_transpiled_prompt_creation(mock_model):
    tp = TranspiledPrompt(
        prompt="Hello transpiled",
        model=mock_model,
        response_format={"type": "json"},
        response="Hi transpiled",
    )
    assert tp.prompt == "Hello transpiled"
    assert tp.model == mock_model


def test_transpiled_prompt_schema(mock_model):
    schema = TranspiledPromptSchema()
    data = {
        "prompt": "Hello transpiled",
        "model": {
            "provider": {"provider": "OpenAI", "provider_type": "api", "metadata": {}},
            "model_name": "gpt-4",
            "supports_system_messages": True,
            "context_window_size": 8192,
            "prompt_style": "markdown",
            "supports_json_mode": True,
            "prompting_tips": "Be concise.",
            "metadata": {},
        },
        "response_format": {"type": "json"},
        "response": "Hi transpiled",
    }
    tp = schema.load(data)
    assert isinstance(tp, TranspiledPrompt)
    assert tp.prompt == "Hello transpiled"
