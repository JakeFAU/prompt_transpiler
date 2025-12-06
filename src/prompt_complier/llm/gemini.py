"""
Gemini LLM Provider implementation.

This module contains the `GeminiAdapter`, which implements the `LLMProvider` interface
for interacting with Google's Gemini API.
"""

from typing import Any

from google import genai
from google.genai import types

from prompt_complier.config import settings
from prompt_complier.utils.logging import get_logger

from .base import LLMProvider

logger = get_logger(__name__)


class GeminiAdapter(LLMProvider):
    """
    Adapter for the Google Gemini API.

    Handles initialization of the async Gemini client and standardized generation calls.
    """

    _client: genai.Client

    def __init__(self) -> None:
        logger.debug("Initializing GeminiAdapter")
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
        config: dict[str, Any],
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generates a response from a Gemini model.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user input.
            model_name: The name of the Gemini model (e.g., "gemini-1.5-flash").
            config: Configuration parameters (e.g., temperature).
            response_schema: Optional JSON schema for structured outputs.
            **kwargs: Additional arguments passed to the Gemini API.

        Returns:
            str: The generated content.
        """
        logger.info("Generating response", provider="gemini", model=model_name)

        # Prepare generation config
        # Map common config keys to Gemini specific keys
        if "max_tokens" in config:
            config["max_output_tokens"] = config.pop("max_tokens")

        gen_config_args = {
            "temperature": settings.GEMINI.TEMPERATURE,
            "system_instruction": system_prompt,
            **config,
            **kwargs,
        }

        # Handle Structured Outputs
        if response_schema:
            logger.debug("Enforcing JSON Schema strict mode")
            gen_config_args["response_mime_type"] = "application/json"
            gen_config_args["response_schema"] = response_schema

        gen_config = types.GenerateContentConfig(**gen_config_args)

        response = await self._client.aio.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=gen_config,
        )

        logger.debug(
            "Gemini generation complete",
            usage=response.usage_metadata.model_dump() if response.usage_metadata else None,
        )

        return response.text or ""

    async def available_models(self) -> list[str]:
        """Fetches available Gemini models."""
        # Note: google-genai SDK 'models.list' might act differently.
        # We'll use the basic list functionality if available or fallback
        # to a hardcoded list if the SDK is restrictive.
        # Checking SDK capabilities:
        pager = await self._client.aio.models.list()

        # Filter for gemini models
        model_names: list[str] = []
        async for m in pager:
            if m.name is not None and "gemini" in m.name:
                model_names.append(m.name)

        # Clean up model names if they come with "models/" prefix
        model_names = [m.replace("models/", "") for m in model_names]

        logger.info("Fetched Gemini models", count=len(model_names))
        return sorted(model_names, reverse=True)
