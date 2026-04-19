"""
Utilities for registering and retrieving supported language models.

`ModelRegistry` seeds a curated list of default models, converts lightweight
dictionary configurations into fully typed `Model` instances, and provides a
simple lookup API with a guarded fallback path for unknown models.
"""

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
        self._register_default_models()
        self._register_huggingface_aliases()

    def _register_default_models(self) -> None:
        """
        Populate the registry with a curated set of known models.

        This list should be revisited periodically to reflect newly released
        models or provider-specific best defaults.
        """
        # OpenAI Models (curated list)
        self.register_model_from_dict(
            {
                "model_name": "gpt-5.2",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 200000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-5.1",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 200000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-5",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 200000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-5-pro",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 200000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-5-mini",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-5-nano",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 64000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-4.1",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-4.1-mini",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-4.1-nano",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 64000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-4o",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gpt-4o-mini",
                "provider": {
                    "provider": "openai",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 64000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )

        # Gemini Models
        self.register_model_from_dict(
            {
                "model_name": "gemini-3-flash-preview",
                "provider": {
                    "provider": "gemini",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 2000000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gemini-3-pro-preview",
                "provider": {
                    "provider": "gemini",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 2000000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gemini-2.5-flash",
                "provider": {
                    "provider": "gemini",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 1000000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gemini-2.5-flash-lite",
                "provider": {
                    "provider": "gemini",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 1000000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gemini-2.5-pro",
                "provider": {
                    "provider": "gemini",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 2000000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gemini-2.0-flash",
                "provider": {
                    "provider": "gemini",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 1000000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "gemini-2.0-flash-lite",
                "provider": {
                    "provider": "gemini",
                    "provider_type": "api",
                },
                "supports_system_messages": True,
                "context_window_size": 1000000,
                "prompt_style": "markdown",
                "supports_json_mode": True,
                "prompting_tips": "Be concise. Use Markdown.",
            }
        )

        # Hugging Face Models (popular models available via Inference API)
        # Meta Llama models
        self.register_model_from_dict(
            {
                "model_name": "meta-llama/Llama-3.3-70B-Instruct",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be clear and direct. Use structured formatting.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be clear and direct. Use structured formatting.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "meta-llama/Llama-3.2-1B-Instruct",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be clear and direct. Use structured formatting.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be clear and direct. Use structured formatting.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "meta-llama/Llama-3.1-70B-Instruct",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be clear and direct. Use structured formatting.",
            }
        )

        # Mistral models
        self.register_model_from_dict(
            {
                "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 32768,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Use clear instructions. Responds well to structure.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 32768,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Use clear instructions. Excels at complex reasoning.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "mistralai/Mistral-Nemo-Instruct-2407",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Use clear instructions. Good for multilingual tasks.",
            }
        )

        # Microsoft Phi models
        self.register_model_from_dict(
            {
                "model_name": "microsoft/Phi-3.5-mini-instruct",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 128000,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be concise. Phi excels at reasoning and code tasks.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "microsoft/Phi-4",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 16384,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be concise. Phi-4 excels at reasoning, math, and coding.",
            }
        )

        # Qwen models
        self.register_model_from_dict(
            {
                "model_name": "Qwen/Qwen2.5-72B-Instruct",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 131072,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be clear and structured. Handles complex instructions.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 131072,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be clear and structured. Handles complex instructions.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 131072,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Specialized for coding tasks. Provide clear code context.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "Qwen/QwQ-32B",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 131072,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Reasoning-focused model. Good for step-by-step problem solving.",
            }
        )

        # DeepSeek models
        self.register_model_from_dict(
            {
                "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 131072,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Strong reasoning model. Good for complex analytical tasks.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 131072,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Strong reasoning model. Good for complex analytical tasks.",
            }
        )

        # Google Gemma models
        self.register_model_from_dict(
            {
                "model_name": "google/gemma-2-27b-it",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 8192,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be concise. Gemma works well with clear, direct instructions.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "google/gemma-2-9b-it",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 8192,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be concise. Gemma works well with clear, direct instructions.",
            }
        )
        self.register_model_from_dict(
            {
                "model_name": "google/gemma-2-2b-it",
                "provider": {
                    "provider": "huggingface",
                    "provider_type": "huggingface",
                },
                "supports_system_messages": True,
                "context_window_size": 8192,
                "prompt_style": "markdown",
                "supports_json_mode": False,
                "prompting_tips": "Be concise. Lightweight model for simple tasks.",
            }
        )

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
            context_window_size=8192,  # Default safe assumption
            prompt_style=style,
            supports_json_mode=True,
            prompting_tips="Be concise.",
        )
