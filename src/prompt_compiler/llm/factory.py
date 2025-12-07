from prompt_compiler.llm.anthropic import AnthropicAdapter
from prompt_compiler.llm.base import LLMProvider
from prompt_compiler.llm.gemini import GeminiAdapter
from prompt_compiler.llm.huggingface import HuggingFaceAdapter
from prompt_compiler.llm.openai import OpenAIAdapter


def get_llm_provider(provider_name: str) -> LLMProvider:
    """
    Factory function to get an LLM provider instance based on the provider name.

    Args:
        provider_name: The name of the provider (e.g., 'openai', 'gemini', 'anthropic').

    Returns:
        An instance of the requested LLMProvider.

    Raises:
        ValueError: If the provider name is not supported.
    """
    normalized_name = provider_name.lower().strip()

    if normalized_name == "openai":
        return OpenAIAdapter()
    elif normalized_name == "gemini":
        return GeminiAdapter()
    elif normalized_name == "anthropic":
        return AnthropicAdapter()
    elif normalized_name == "huggingface":
        return HuggingFaceAdapter()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
