from unittest.mock import patch

import pytest

from prompt_compiler.llm.anthropic import AnthropicAdapter
from prompt_compiler.llm.factory import get_llm_provider
from prompt_compiler.llm.gemini import GeminiAdapter
from prompt_compiler.llm.huggingface import HuggingFaceAdapter
from prompt_compiler.llm.openai import OpenAIAdapter


@patch("prompt_compiler.llm.openai.settings")
def test_get_llm_provider_openai(mock_settings):
    mock_settings.OPENAI_API_KEY = "test-openai-key"

    provider = get_llm_provider("openai")
    assert isinstance(provider, OpenAIAdapter)

    provider = get_llm_provider("OPENAI")
    assert isinstance(provider, OpenAIAdapter)


@patch("prompt_compiler.llm.gemini.settings")
def test_get_llm_provider_gemini(mock_settings):
    mock_settings.GEMINI_API_KEY = "test-gemini-key"

    provider = get_llm_provider("gemini")
    assert isinstance(provider, GeminiAdapter)


@patch("prompt_compiler.llm.anthropic.settings")
def test_get_llm_provider_anthropic(mock_settings):
    mock_settings.ANTHROPIC_API_KEY = "test-anthropic-key"

    provider = get_llm_provider("anthropic")
    assert isinstance(provider, AnthropicAdapter)


@patch("prompt_compiler.llm.huggingface.settings")
def test_get_llm_provider_huggingface(mock_settings):
    mock_settings.HUGGINGFACE_API_KEY = "test-huggingface-key"

    provider = get_llm_provider("huggingface")
    assert isinstance(provider, HuggingFaceAdapter)


def test_get_llm_provider_invalid():
    with pytest.raises(ValueError, match="Unsupported LLM provider: invalid"):
        get_llm_provider("invalid")
