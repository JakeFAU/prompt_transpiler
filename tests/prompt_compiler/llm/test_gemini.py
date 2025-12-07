from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_compiler.llm.gemini import GeminiAdapter


@pytest.fixture(autouse=True)
def mock_settings(mocker):
    mock_conf = mocker.patch("prompt_compiler.llm.gemini.settings")
    mock_conf.GEMINI_API_KEY = "sk-gemini-key"
    mock_conf.GEMINI.TEMPERATURE = 0.0
    return mock_conf


@pytest.fixture
def mock_gemini(mocker):
    return mocker.patch("prompt_compiler.llm.gemini.genai.Client")


@pytest.mark.asyncio
async def test_generate_simple(mock_gemini):
    """Test simple text generation for Gemini."""
    mock_client_instance = mock_gemini.return_value

    # Mock the response object
    mock_response = MagicMock()
    mock_response.text = "Gemini Response"
    mock_response.usage_metadata.model_dump.return_value = {"total_tokens": 10}

    # Mock the async generate_content method
    mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)

    adapter = GeminiAdapter()
    response = await adapter.generate(
        system_prompt="System",
        user_prompt="User",
        model_name="gemini-1.5-flash",
        config={"max_tokens": 100},
    )

    assert response == "Gemini Response"
    mock_client_instance.aio.models.generate_content.assert_called_once()

    call_kwargs = mock_client_instance.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-1.5-flash"
    assert call_kwargs["contents"] == "User"

    config_arg = call_kwargs["config"]
    assert config_arg.temperature == 0.0
    assert config_arg.system_instruction == "System"
    assert config_arg.max_output_tokens == 100  # noqa: PLR2004


@pytest.mark.asyncio
async def test_generate_with_schema(mock_gemini):
    """Test generation with JSON schema enforcement for Gemini."""
    mock_client_instance = mock_gemini.return_value

    mock_response = MagicMock()
    mock_response.text = "{}"
    mock_client_instance.aio.models.generate_content = AsyncMock(return_value=mock_response)

    adapter = GeminiAdapter()
    response_schema = {"type": "object", "properties": {"foo": {"type": "string"}}}

    await adapter.generate(
        system_prompt="System",
        user_prompt="User",
        model_name="gemini-1.5-flash",
        config={},
        response_schema=response_schema,
    )

    call_kwargs = mock_client_instance.aio.models.generate_content.call_args.kwargs
    config_arg = call_kwargs["config"]

    assert config_arg.response_mime_type == "application/json"
    assert config_arg.response_schema == response_schema


@pytest.mark.asyncio
async def test_available_models(mock_gemini):
    """Test fetching available Gemini models."""
    mock_client_instance = mock_gemini.return_value

    mock_model1 = MagicMock()
    mock_model1.name = "models/gemini-1.5-flash"
    mock_model2 = MagicMock()
    mock_model2.name = "models/gemini-1.0-pro"
    mock_model3 = MagicMock()
    # Should be filtered out if logic was stricter, but we filter by "gemini" in name
    mock_model3.name = "models/embedding-001"

    # Create an async iterator for the mock list
    async def async_gen() -> AsyncIterator[MagicMock]:
        for m in [mock_model1, mock_model2]:
            yield m

    mock_client_instance.aio.models.list = AsyncMock(return_value=async_gen())

    adapter = GeminiAdapter()
    models = await adapter.available_models()

    assert "gemini-1.5-flash" in models
    assert "gemini-1.0-pro" in models
