"""
Anthropic LLM Provider implementation.

This module contains the `AnthropicAdapter`, which implements the `LLMProvider` interface
for interacting with Anthropic's API.
"""

import json
from typing import Any

from anthropic import AsyncAnthropic

from prompt_compiler.config import settings
from prompt_compiler.utils.logging import get_logger

from .base import LLMProvider

logger = get_logger(__name__)


class AnthropicAdapter(LLMProvider):
    """
    Adapter for the Anthropic API.

    Handles initialization of the async Anthropic client and standardized generation calls.
    """

    _client: AsyncAnthropic

    def __init__(self) -> None:
        logger.debug("Initializing AnthropicAdapter")
        self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

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
        Generates a response from an Anthropic model.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user input.
            model_name: The name of the Anthropic model (e.g., "claude-3-opus").
            config: Configuration parameters (e.g., temperature).
            response_schema: Optional JSON schema for structured outputs.
            **kwargs: Additional arguments passed to the Anthropic API.

        Returns:
            str: The generated content.
        """
        logger.info("Generating response", provider="anthropic", model=model_name)

        # Merge defaults
        params = {
            "model": model_name,
            "temperature": settings.ANTHROPIC.TEMPERATURE,
            "max_tokens": 4096,  # Anthropic requires max_tokens to be set
            **config,
            **kwargs,
        }

        # Handle Structured Outputs (Simulated via Prompt Engineering for now)
        # Note: As of late 2024/2025, Anthropic might have better JSON support,
        # but prompt augmentation is the fallback.
        final_system_prompt = system_prompt
        if response_schema:
            logger.debug("Appending JSON Schema to system prompt for Anthropic")

            schema_str = json.dumps(response_schema, indent=2)
            final_system_prompt += (
                "\n\nYou must output valid JSON strictly adhering to the following schema:\n"
                f"{schema_str}"
            )
            # Force the assistant to start with { to encourage JSON
            # We won't do it here as it complicates the 'messages' construction if not careful.

        messages = [
            {"role": "user", "content": user_prompt},
        ]

        response = await self._client.messages.create(
            system=final_system_prompt,
            messages=messages,  # type: ignore[arg-type]
            **params,
        )

        logger.debug(
            "Anthropic generation complete",
            id=response.id,
            usage=response.usage.model_dump() if response.usage else None,
        )

        # Extract text content
        content = ""
        if response.content:
            for block in response.content:
                if block.type == "text":
                    content += block.text

        return content

    async def available_models(self) -> list[str]:
        """Returns a hardcoded list of Anthropic models as the API doesn't support listing."""
        # Anthropic API does not support listing models programmatically as of v0.x
        # We return a standard list of known models.
        models = [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
        logger.info("Returning hardcoded Anthropic models", count=len(models))
        return models
