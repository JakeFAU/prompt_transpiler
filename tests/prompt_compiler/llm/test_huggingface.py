from unittest.mock import AsyncMock, MagicMock

import pytest

from prompt_complier.llm.huggingface import HuggingFaceAdapter


@pytest.fixture(autouse=True)
def mock_settings(mocker):
    mock_conf = mocker.patch("prompt_complier.llm.huggingface.settings")
    mock_conf.HUGGINGFACE_API_KEY = "hf_token"
    mock_conf.HUGGINGFACE.TEMPERATURE = 0.0
    return mock_conf


@pytest.fixture
def mock_hf_client(mocker):
    return mocker.patch("prompt_complier.llm.huggingface.AsyncInferenceClient")


@pytest.fixture
def mock_list_models(mocker):
    return mocker.patch("prompt_complier.llm.huggingface.list_models")


@pytest.mark.asyncio
async def test_generate_simple(mock_hf_client):
    """Test simple text generation for Hugging Face."""
    mock_client_instance = mock_hf_client.return_value

    # Mock response
    mock_choice = MagicMock()
    mock_choice.message.content = "HF Response"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage.model_dump.return_value = {"total_tokens": 10}

    mock_client_instance.chat_completion = AsyncMock(return_value=mock_completion)

    adapter = HuggingFaceAdapter()
    response = await adapter.generate(
        system_prompt="System",
        user_prompt="User",
        model_name="meta-llama/Llama-2-7b",
        config={"max_tokens": 100},
    )

    assert response == "HF Response"
    mock_client_instance.chat_completion.assert_called_once()

    call_kwargs = mock_client_instance.chat_completion.call_args.kwargs
    assert call_kwargs["model"] == "meta-llama/Llama-2-7b"
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User"},
    ]


@pytest.mark.asyncio
async def test_generate_with_schema(mock_hf_client):
    """Test generation with schema (prompt augmentation)."""
    mock_client_instance = mock_hf_client.return_value
    mock_choice = MagicMock()
    mock_choice.message.content = "{}"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_client_instance.chat_completion = AsyncMock(return_value=mock_completion)

    adapter = HuggingFaceAdapter()
    response_schema = {"type": "object"}

    await adapter.generate(
        system_prompt="System",
        user_prompt="User",
        model_name="meta-llama/Llama-2-7b",
        config={},
        response_schema=response_schema,
    )

    call_kwargs = mock_client_instance.chat_completion.call_args.kwargs
    # Check if system prompt was modified in the messages
    messages = call_kwargs["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert "output valid JSON" in system_msg["content"]


@pytest.mark.asyncio
async def test_available_models(mock_hf_client, mock_list_models):
    """Test fetching available HF models."""
    mock_model1 = MagicMock(id="model1")
    mock_model2 = MagicMock(id="model2")
    mock_list_models.return_value = [mock_model1, mock_model2]

    adapter = HuggingFaceAdapter()
    models = await adapter.available_models()

    assert "model1" in models
    assert "model2" in models
    mock_list_models.assert_called_once()
