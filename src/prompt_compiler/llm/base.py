"""
Base classes and interfaces for Large Language Model (LLM) providers.

This module defines the `LLMProvider` abstract base class, which standardizes
interactions with different LLM backends (e.g., OpenAI, Gemini, Anthropic).
It also provides utility functions for discovering available providers.
"""

from abc import ABC, abstractmethod
from typing import Any

from prompt_compiler.config import settings


class LLMProvider(ABC):
    """
    Abstract Base Class for all LLM backends.
    """

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        config: dict[str, Any],
        # Crucial for the Decompiler to enforce valid IR generation
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Standardized generation method.

        Args:
            system_prompt: The system instruction (Compiler Persona).
            user_prompt: The user input or prompt to be compiled.
            model_name: The specific model to use for generation.
            config: Generation params (temp, max_tokens).
            response_schema: A JSON schema dict to enforce structured output.
            **kwargs: Provider-specific arguments (e.g. top_k, seed).
        """
        pass

    @abstractmethod
    async def available_models(self) -> list[str]:
        """Returns a list of available models for the provider."""
        pass


def available_llm_providers() -> list[str]:
    """Returns a list of available LLM providers based on env vars."""
    p: list[str] = []
    if settings.OPENAI_API_KEY:
        p.append("openai")
    if settings.GEMINI_API_KEY:
        p.append("gemini")
    if settings.ANTHROPIC_API_KEY:
        p.append("anthropic")
    if settings.HUGGINGFACE_API_KEY:
        p.append("huggingface")
    return p
