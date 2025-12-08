"""
OpenAI LLM Provider implementation.

This module contains the `OpenAIAdapter`, which implements the `LLMProvider` interface
for interacting with OpenAI's chat completion API.
"""

from copy import deepcopy
from typing import Any

import openai
from openai.types.chat.chat_completion import ChatCompletion

from prompt_compiler.config import settings
from prompt_compiler.utils.logging import get_logger

from .base import LLMProvider

logger = get_logger(__name__)


def _prepare_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    OpenAI strict JSON schema requires:
    - additionalProperties: false on every object
    - required must include every property key (no extras)
    This helper walks the schema, enforces those rules, and returns a new copy so callers
    are not mutated.
    """

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                if node.get("additionalProperties") is not False:
                    node["additionalProperties"] = False

                if "properties" in node:
                    node["required"] = list(node["properties"].keys())

            for key in ("properties", "patternProperties", "$defs", "definitions"):
                for child in (node.get(key) or {}).values():
                    _walk(child)

            if "items" in node:
                _walk(node["items"])

            for key in ("anyOf", "oneOf", "allOf"):
                for child in node.get(key, []):
                    _walk(child)

        elif isinstance(node, list):
            for item in node:
                _walk(item)

    schema_copy = deepcopy(schema)
    _walk(schema_copy)
    return schema_copy


class OpenAIAdapter(LLMProvider):
    """
    Adapter for the OpenAI API.

    Handles initialization of the async OpenAI client and standardized generation calls.
    """

    _client: openai.AsyncOpenAI

    def __init__(self) -> None:
        logger.debug("Initializing OpenAIAdapter")
        self._client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        # Note: Ensure your ABC 'generate' signature matches this,
        # or grab model_name from **config if the ABC is strict.
        model_name: str,
        config: dict[str, Any],
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generates a response from an OpenAI model.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user input.
            model_name: The name of the OpenAI model (e.g., "gpt-4").
            config: Configuration parameters (e.g., temperature).
            response_schema: Optional JSON schema for structured outputs.
            **kwargs: Additional arguments passed to the OpenAI API.

        Returns:
            str: The generated content.
        """
        logger.info("Generating response", provider="openai", model=model_name)

        # Merge defaults
        params = {
            "model": model_name,
            "temperature": settings.OPENAI.TEMPERATURE,
            **config,
            **kwargs,
        }

        # Handle Structured Outputs
        if response_schema:
            logger.debug("Enforcing JSON Schema strict mode")
            safe_schema = _prepare_strict_schema(response_schema)
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ir_response", "strict": True, "schema": safe_schema},
            }
        elif params.get("response_format") == {"type": "json_object"}:
            logger.debug("Using legacy JSON mode")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # The Fix: Ignore the overload error caused by **params unpacking
        response: ChatCompletion = await self._client.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            **params,
        )
        # Log metadata, not content
        logger.debug(
            "OpenAI generation complete",
            id=response.id,
            usage=response.usage.model_dump() if response.usage else None,
        )

        return response.choices[0].message.content or ""

    async def available_models(self) -> list[str]:
        """Fetches available GPT models (filtering out audio/image models)."""
        pager = await self._client.models.list()

        # Simple filter to keep the list relevant for a Prompt Compiler
        model_names = [m.id for m in pager.data if "gpt" in m.id or "o1" in m.id]

        logger.info("Fetched OpenAI models", count=len(model_names))
        return sorted(model_names, reverse=True)
