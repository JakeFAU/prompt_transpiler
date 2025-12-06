from attrs import define

from prompt_complier.core.exceptions import ProviderError
from prompt_complier.core.interfaces import IHistorian
from prompt_complier.llm.anthropic import AnthropicAdapter
from prompt_complier.llm.base import LLMProvider
from prompt_complier.llm.gemini import GeminiAdapter
from prompt_complier.llm.huggingface import HuggingFaceAdapter
from prompt_complier.llm.openai import OpenAIAdapter
from prompt_complier.llm.prompts.prompt_objects import OriginalPrompt
from prompt_complier.utils.logging import get_logger
from prompt_complier.utils.telemetry import telemetry

logger = get_logger(__name__)


def get_provider_adapter(provider_name: str) -> LLMProvider:
    """Factory to get the correct LLMProvider adapter."""
    # TODO: Move this to a proper Factory module/class
    provider_name = provider_name.lower()
    if "openai" in provider_name:
        return OpenAIAdapter()
    elif "gemini" in provider_name or "google" in provider_name:
        return GeminiAdapter()
    elif "anthropic" in provider_name:
        return AnthropicAdapter()
    elif "huggingface" in provider_name:
        return HuggingFaceAdapter()
    else:
        logger.warning("Unknown provider, defaulting to OpenAI", provider=provider_name)
        return OpenAIAdapter()


@define
class DefaultHistorian(IHistorian):
    """
    Standard implementation of the Historian role.

    The Historian's responsibility is to benchmark the "Original Prompt" against
    its native model. This provides a baseline response to which the "Candidate Prompt"
    (running on the new target model) can be compared by the Judge.
    """

    @telemetry.instrument(name="historian.establish_baseline")
    async def establish_baseline(self, original_prompt: OriginalPrompt) -> OriginalPrompt:
        """
        Execute the original prompt on its source model to capture a baseline response.

        Args:
            original_prompt: The prompt object with raw text and source model metadata.

        Returns:
            OriginalPrompt: The same object, mutated to include the 'response' string
                            from the model.

        Raises:
            ProviderError: If the LLM provider fails to generate a response.
        """
        logger.info("Historian starting baseline run", model=original_prompt.model.model_name)

        provider = get_provider_adapter(original_prompt.model.provider.provider)

        try:
            # We use a deterministic config for the baseline to be stable
            response = await provider.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt=original_prompt.prompt,
                model_name=original_prompt.model.model_name,
                config={"temperature": 0.0},
            )
            original_prompt.response = response
            logger.info("Historian baseline captured", response_length=len(response))

        except Exception as e:
            logger.error("Historian failed to run baseline", error=str(e))
            # Depending on policy, we might want to raise or allow empty baseline.
            # For a robust compiler, missing baseline is critical, so we raise.
            raise ProviderError(
                f"Failed to get baseline from {original_prompt.model.model_name}"
            ) from e

        return original_prompt
