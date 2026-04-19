"""
Utilities for registering and retrieving supported language models.

`ModelRegistry` seeds a curated list of default models, converts lightweight
dictionary configurations into fully typed `Model` instances, and provides a
simple lookup API with a guarded fallback path for unknown models.
"""

import json
from importlib import resources
from typing import Any

from attrs import define, field

from prompt_transpiler.dto.models import Model, ModelProviderType, PromptStyle, Provider
from prompt_transpiler.utils.logging import get_logger

logger = get_logger(__name__)


@define
class ModelRegistry:
    """
    Central registry for the models the Prompt Transpiler can target.

    The registry:
    - Bootstraps with a set of common models from major providers.
    - Accepts dynamic registrations from dictionary-based configs.
    - Exposes a lookup method that can synthesize a temporary model definition
      when a requested model is missing, minimizing disruptions for callers.
    """

    _models: dict[str, Model] = field(factory=dict)
    _aliases: dict[str, str] = field(factory=dict)

    def __attrs_post_init__(self) -> None:
        """Initialize the registry with default models on construction."""
        self._load_models_from_json()
        self._register_huggingface_aliases()

    def _load_models_from_json(self) -> None:
        """Load model definitions from the bundled models.json file."""
        try:
            # use importlib.resources to load the models.json file
            # relative to the package
            pkg = "prompt_transpiler.core"
            with resources.files(pkg).joinpath("models.json").open("r") as f:
                models_data = json.load(f)
                for model_dict in models_data:
                    self.register_model_from_dict(model_dict)
            logger.debug("Successfully loaded models from models.json")
        except Exception as e:
            logger.error("Failed to load models.json: %s", e)
            # Fallback to empty registry if loading fails
            # We could also re-inject a few bare minimums here if needed

    def _register_huggingface_aliases(self) -> None:
        """
        Register short-name aliases for HuggingFace models.

        This allows users to specify just the model name (e.g., 'Llama-3.3-70B-Instruct')
        instead of the full HuggingFace model ID (e.g., 'meta-llama/Llama-3.3-70B-Instruct').
        """
        # Map short names to full HuggingFace model IDs
        hf_aliases = {
            # Meta Llama
            "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
            "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
            "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
            "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
            "Llama-3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
            # Mistral
            "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
            "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "Mistral-Nemo-Instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407",
            # Microsoft Phi
            "Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
            "Phi-4": "microsoft/Phi-4",
            # Qwen
            "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
            "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
            "Qwen2.5-Coder-32B-Instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "QwQ-32B": "Qwen/QwQ-32B",
            # DeepSeek
            "DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "DeepSeek-R1-Distill-Llama-70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            # Google Gemma
            "gemma-2-27b-it": "google/gemma-2-27b-it",
            "gemma-2-9b-it": "google/gemma-2-9b-it",
            "gemma-2-2b-it": "google/gemma-2-2b-it",
        }
        self._aliases.update(hf_aliases)
        logger.debug("Registered %d HuggingFace model aliases", len(hf_aliases))

    def register_model(self, model: Model) -> None:
        """
        Register a fully constructed `Model` instance.

        Args:
            model: The model definition to add to the registry. Existing entries
                with the same `model_name` are overwritten.
        """
        self._models[model.model_name] = model
        logger.debug(f"Registered model: {model.model_name}")

    def register_model_from_dict(self, data: dict[str, Any]) -> Model:
        """
        Register a model from a plain dictionary configuration.

        The method performs light normalization, including conversion of provider
        and prompt style strings to their enum counterparts, before constructing
        the `Model`.

        Args:
            data: Model configuration containing keys such as `model_name`,
                `provider`, `prompt_style`, and other `Model` fields. Nested
                provider data can be provided as a dictionary.

        Returns:
            The `Model` instance that was registered.
        """
        # Handle Provider creation
        provider_data = data.get("provider")
        if isinstance(provider_data, dict):
            # Ensure provider_type is an Enum member if it's a string
            if isinstance(provider_data.get("provider_type"), str):
                try:
                    provider_data["provider_type"] = ModelProviderType(
                        provider_data["provider_type"]
                    )
                except ValueError:
                    # Fallback or error handling
                    logger.warning(
                        "Unknown provider type: %s, defaulting to API",
                        provider_data.get("provider_type"),
                    )
                    provider_data["provider_type"] = ModelProviderType.API

            data["provider"] = Provider(**provider_data)

        # Handle PromptStyle enum conversion
        if isinstance(data.get("prompt_style"), str):
            try:
                data["prompt_style"] = PromptStyle(data["prompt_style"])
            except ValueError:
                logger.warning(
                    "Unknown prompt style: %s, defaulting to MARKDOWN",
                    data.get("prompt_style"),
                )
                data["prompt_style"] = PromptStyle.MARKDOWN

        model = Model(**data)
        self.register_model(model)
        return model

    def get_model(self, model_name: str, provider_name: str | None = None) -> Model:
        """
        Retrieve a model by name, with an optional provider hint.

        Args:
            model_name: Canonical name of the model to retrieve.
            provider_name: Optional provider identifier used when constructing a
                temporary fallback model.

        Returns:
            The matching `Model` from the registry, or a temporary definition
            synthesized with conservative defaults when the name is unknown.
        """
        # Direct lookup
        if model_name in self._models:
            return self._models[model_name]

        # Check if it's an alias (e.g., short name for HuggingFace models)
        if model_name in self._aliases:
            canonical_name = self._aliases[model_name]
            logger.debug(
                "Resolved alias '%s' to canonical model '%s'",
                model_name,
                canonical_name,
            )
            return self._models[canonical_name]

        # Soft matching or fallback logic can go here.
        # For now, let's try to handle the case where user provides "gpt-4" but we
        # have "gpt-4" registered. But if they provide "claude" and we don't have
        # it, we might want to throw or return a generic one.

        # Since the prompt said "It should also have a way of reading in a new model...",
        # I'll assume if it's not in the registry, we might need to fallback to the old behavior
        # OR just fail.
        # However, to avoid breaking the current experience entirely if they use an unknown model,
        # I will re-implement a robust fallback that warns.

        logger.warning(
            "Model '%s' not found in registry. Creating temporary model definition.",
            model_name,
        )

        # Fallback logic similar to the old create_dummy_model to ensure continuity
        p_type = ModelProviderType.API
        normalized_provider = (provider_name or "unknown").lower().strip()
        if "huggingface" in normalized_provider:
            p_type = ModelProviderType.HUGGINGFACE

        style = PromptStyle.MARKDOWN
        if "claude" in model_name.lower() or "anthropic" in normalized_provider:
            style = PromptStyle.XML

        return Model(
            provider=Provider(
                provider=normalized_provider,
                provider_type=p_type,
            ),
            model_name=model_name,
            supports_system_messages=True,
            supports_system_instructions=True,
            supports_structured_outputs=False,
            context_window_size=8192,  # Default safe assumption
            prompt_style=style,
            supports_json_mode=True,
            prompting_tips="Be concise.",
        )
