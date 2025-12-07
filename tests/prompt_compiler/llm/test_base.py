from unittest.mock import patch

import pytest

from prompt_compiler.llm.base import LLMProvider, available_llm_providers

# Test constants
EXPECTED_PROVIDER_COUNT = 4


def test_available_llm_providers_all_set():
    with patch("prompt_compiler.llm.base.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = "sk-..."
        mock_settings.GEMINI_API_KEY = "AI..."
        mock_settings.ANTHROPIC_API_KEY = "sk-ant..."
        mock_settings.HUGGINGFACE_API_KEY = "hf_..."

        providers = available_llm_providers()
        assert "openai" in providers
        assert "gemini" in providers
        assert "anthropic" in providers
        assert "huggingface" in providers
        assert len(providers) == EXPECTED_PROVIDER_COUNT


def test_available_llm_providers_none_set():
    with patch("prompt_compiler.llm.base.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = None
        mock_settings.GEMINI_API_KEY = None
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.HUGGINGFACE_API_KEY = None

        providers = available_llm_providers()
        assert len(providers) == 0


def test_available_llm_providers_partial():
    with patch("prompt_compiler.llm.base.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = "sk-..."
        mock_settings.GEMINI_API_KEY = None
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.HUGGINGFACE_API_KEY = None

        providers = available_llm_providers()
        assert "openai" in providers
        assert len(providers) == 1


def test_llm_provider_abstract():
    # Ensure LLMProvider cannot be instantiated directly
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]
