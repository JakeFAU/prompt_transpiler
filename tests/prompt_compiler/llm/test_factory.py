import pytest

from prompt_complier.llm.anthropic import AnthropicAdapter
from prompt_complier.llm.factory import get_llm_provider
from prompt_complier.llm.gemini import GeminiAdapter
from prompt_complier.llm.huggingface import HuggingFaceAdapter
from prompt_complier.llm.openai import OpenAIAdapter


def test_get_llm_provider_openai():
    provider = get_llm_provider("openai")
    assert isinstance(provider, OpenAIAdapter)

    provider = get_llm_provider("OPENAI")
    assert isinstance(provider, OpenAIAdapter)


def test_get_llm_provider_gemini():
    provider = get_llm_provider("gemini")
    assert isinstance(provider, GeminiAdapter)


def test_get_llm_provider_anthropic():
    provider = get_llm_provider("anthropic")
    assert isinstance(provider, AnthropicAdapter)


def test_get_llm_provider_huggingface():
    provider = get_llm_provider("huggingface")
    assert isinstance(provider, HuggingFaceAdapter)


def test_get_llm_provider_invalid():
    with pytest.raises(ValueError, match="Unsupported LLM provider: invalid"):
        get_llm_provider("invalid")
