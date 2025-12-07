from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_compiler.llm.anthropic import AnthropicAdapter


@pytest.fixture(autouse=True)
def mock_settings(mocker):
    mock_conf = mocker.patch("prompt_compiler.llm.anthropic.settings")
    mock_conf.ANTHROPIC_API_KEY = "sk-ant-key"
    mock_conf.ANTHROPIC.TEMPERATURE = 0.0
    return mock_conf


@pytest.fixture
def mock_anthropic(mocker):
    return mocker.patch("prompt_compiler.llm.anthropic.AsyncAnthropic")


@pytest.mark.asyncio
async def test_generate_simple(mock_anthropic):
    """Test simple text generation for Anthropic."""
    mock_client_instance = mock_anthropic.return_value

    # Mock response
    mock_msg = MagicMock()
    mock_msg.id = "msg_123"
    mock_msg.content = [MagicMock(type="text", text="Anthropic Response")]
    mock_msg.usage.model_dump.return_value = {"input_tokens": 10, "output_tokens": 5}

    mock_client_instance.messages.create = AsyncMock(return_value=mock_msg)

    adapter = AnthropicAdapter()
    response = await adapter.generate(
        system_prompt="System",
        user_prompt="User",
        model_name="claude-3-opus",
        config={"max_tokens": 100},
    )

    assert response == "Anthropic Response"
    mock_client_instance.messages.create.assert_called_once()

    call_kwargs = mock_client_instance.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-3-opus"
    assert call_kwargs["system"] == "System"
    assert call_kwargs["messages"] == [{"role": "user", "content": "User"}]
    assert call_kwargs["max_tokens"] == 100  # noqa: PLR2004


@pytest.mark.asyncio
async def test_generate_with_schema(mock_anthropic):
    """Test generation with schema (prompt augmentation)."""
    mock_client_instance = mock_anthropic.return_value
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(type="text", text="{}")]
    mock_client_instance.messages.create = AsyncMock(return_value=mock_msg)

    adapter = AnthropicAdapter()
    response_schema = {"type": "object"}

    await adapter.generate(
        system_prompt="System",
        user_prompt="User",
        model_name="claude-3-opus",
        config={},
        response_schema=response_schema,
    )

    call_kwargs = mock_client_instance.messages.create.call_args.kwargs
    # Check if system prompt was modified
    assert "output valid JSON" in call_kwargs["system"]


@pytest.mark.asyncio
async def test_available_models(mock_anthropic):
    """Test fetching available models (hardcoded)."""
    adapter = AnthropicAdapter()
    models = await adapter.available_models()

    assert len(models) > 0
    assert "claude-3-5-sonnet-latest" in models
