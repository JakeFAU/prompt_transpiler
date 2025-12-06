"""
Hugging Face LLM Provider implementation.

This module contains the `HuggingFaceAdapter`, which implements the `LLMProvider` interface
for interacting with Hugging Face Inference API (Serverless or TGI).
"""

import json
from typing import Any

from huggingface_hub import AsyncInferenceClient, list_models

from prompt_complier.config import settings
from prompt_complier.utils.logging import get_logger

from .base import LLMProvider

logger = get_logger(__name__)


class HuggingFaceAdapter(LLMProvider):
    """
    Adapter for the Hugging Face Inference API.

    Handles initialization of the async client and standardized generation calls.
    """

    _client: AsyncInferenceClient

    def __init__(self) -> None:
        logger.debug("Initializing HuggingFaceAdapter")
        self._client = AsyncInferenceClient(token=settings.HUGGINGFACE_API_KEY)

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
        Generates a response from a Hugging Face model.

        Args:
            system_prompt: The system instruction.
            user_prompt: The user input.
            model_name: The name of the HF model (e.g., "meta-llama/Llama-2-70b-chat-hf").
            config: Configuration parameters (e.g., temperature).
            response_schema: Optional JSON schema for structured outputs.
            **kwargs: Additional arguments passed to the HF API.

        Returns:
            str: The generated content.
        """
        logger.info("Generating response", provider="huggingface", model=model_name)

        # Merge defaults
        params = {
            "model": model_name,
            "temperature": settings.HUGGINGFACE.TEMPERATURE,
            "max_tokens": 1024,  # Default max tokens
            **config,
            **kwargs,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Handle Structured Outputs (Simulated via Prompt Engineering)
        # Note: Some HF endpoints support 'grammar' or 'response_format',
        # but it depends on the backend (TGI, vLLM).
        # We'll use prompt augmentation for broader compatibility unless configured otherwise.
        if response_schema:
            logger.debug("Appending JSON Schema to system prompt for Hugging Face")

            schema_str = json.dumps(response_schema, indent=2)
            # Modify the system prompt in the messages list
            messages[0]["content"] += (
                "\n\nYou must output valid JSON strictly adhering to the following schema:\n"
                f"{schema_str}"
            )

        # Using chat_completion which handles formatting for instruction models
        response = await self._client.chat_completion(
            messages=messages,
            **params,
        )

        logger.debug(
            "Hugging Face generation complete",
            usage=response.usage.model_dump() if response.usage else None,
        )

        return response.choices[0].message.content or ""

    async def available_models(self) -> list[str]:
        """Fetches available text-generation models from the Hub (top 20 by downloads)."""
        # Fetch top models to avoid listing thousands
        try:
            # This is a synchronous call, might want to wrap in run_in_executor
            # if blocking too long, but for this CLI tool it should be fine.
            models = list_models(
                filter="text-generation-inference",  # Models compatible with TGI/Inference API
                sort="downloads",
                direction=-1,
                limit=20,
            )
            model_names = [m.id for m in models]
            logger.info("Fetched Hugging Face models", count=len(model_names))
            return model_names
        except Exception as e:
            logger.error("Failed to fetch HF models", error=str(e))
            return []
