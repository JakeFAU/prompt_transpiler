"""Historian role implementation for baseline capture."""

from attrs import define

from prompt_transpiler.core.exceptions import ProviderError
from prompt_transpiler.core.interfaces import IHistorian
from prompt_transpiler.llm.factory import get_llm_provider
from prompt_transpiler.llm.prompts.prompt_objects import OriginalPrompt
from prompt_transpiler.utils.logging import get_logger
from prompt_transpiler.utils.telemetry import telemetry
from prompt_transpiler.utils.token_collector import token_collector

logger = get_logger(__name__)


@define
class DefaultHistorian(IHistorian):
    """
    Standard implementation of the Historian role.

    The Historian's responsibility is to benchmark the "Original Prompt" against
    its native model.
    """

    @telemetry.instrument(name="historian.establish_baseline")
    async def establish_baseline(self, original_prompt: OriginalPrompt) -> OriginalPrompt:
        """
        Execute the original prompt on its source model to capture a baseline response.
        """
        logger.info("Historian starting baseline run", model=original_prompt.model.model_name)

        provider = get_llm_provider(original_prompt.model.provider.provider)

        try:
            # We use a deterministic config for the baseline to be stable
            llm_response = await provider.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt=original_prompt.prompt,
                model_name=original_prompt.model.model_name,
                config={"temperature": 0.0},
            )
            # Collect tokens
            token_collector.add(original_prompt.model.model_name, llm_response.usage)

            response = llm_response.content
            original_prompt.response = response
            logger.info("Historian baseline captured", response_length=len(response))

        except Exception as e:
            logger.error("Historian failed to run baseline", error=str(e))
            raise ProviderError(
                f"Failed to get baseline from {original_prompt.model.model_name}"
            ) from e

        return original_prompt
