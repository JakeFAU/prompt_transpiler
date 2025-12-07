from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_compiler.llm.openai import OpenAIAdapter


@pytest.fixture(autouse=True)
def mock_settings(mocker):
    mock_conf = mocker.patch("prompt_compiler.llm.openai.settings")
    mock_conf.OPENAI_API_KEY = "sk-test-key"
    mock_conf.OPENAI.TEMPERATURE = 0.0
    return mock_conf


@pytest.fixture
def mock_openai(mocker):
    # Mock the class itself so when instantiated it returns our mock object
    return mocker.patch("prompt_compiler.llm.openai.openai.AsyncOpenAI")


@pytest.mark.asyncio
async def test_generate_simple(mock_openai):
    """Test simple text generation."""
    # Setup the mock client instance
    mock_client_instance = mock_openai.return_value

    # Setup the response structure
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="Generated response"))]
    mock_completion.usage = MagicMock(model_dump=lambda: {"total_tokens": 10})
    mock_completion.id = "test-id"

    # Mock the async create method
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_completion)

    adapter = OpenAIAdapter()
    response = await adapter.generate(
        system_prompt="System", user_prompt="User", model_name="gpt-4", config={"max_tokens": 100}
    )

    assert response == "Generated response"
    mock_client_instance.chat.completions.create.assert_called_once()

    call_kwargs = mock_client_instance.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4"
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User"},
    ]
    assert call_kwargs["max_tokens"] == 100  # noqa: PLR2004


@pytest.mark.asyncio
async def test_generate_with_schema(mock_openai):
    """Test generation with JSON schema enforcement."""
    mock_client_instance = mock_openai.return_value

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="{}"))]
    mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_completion)

    adapter = OpenAIAdapter()
    response_schema = {"type": "object", "properties": {"foo": {"type": "string"}}}

    await adapter.generate(
        system_prompt="System",
        user_prompt="User",
        model_name="gpt-4",
        config={},
        response_schema=response_schema,
    )

    call_kwargs = mock_client_instance.chat.completions.create.call_args.kwargs
    assert "response_format" in call_kwargs
    assert call_kwargs["response_format"]["type"] == "json_schema"
    assert call_kwargs["response_format"]["json_schema"]["schema"] == response_schema


@pytest.mark.asyncio
async def test_available_models(mock_openai):
    """Test fetching available models."""
    mock_client_instance = mock_openai.return_value

    # Mock model list response
    mock_model1 = MagicMock(id="gpt-4")
    mock_model2 = MagicMock(id="gpt-3.5-turbo")
    mock_model3 = MagicMock(id="dall-e-3")  # Should be filtered out

    mock_pager = MagicMock()
    mock_pager.data = [mock_model1, mock_model2, mock_model3]

    mock_client_instance.models.list = AsyncMock(return_value=mock_pager)

    adapter = OpenAIAdapter()
    models = await adapter.available_models()

    assert "gpt-4" in models
    assert "gpt-3.5-turbo" in models
    assert "dall-e-3" not in models
