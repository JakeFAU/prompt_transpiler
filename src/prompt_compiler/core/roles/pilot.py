"""Pilot role implementation for candidate evaluation runs."""

from attrs import define

from prompt_compiler.core.interfaces import IPilot
from prompt_compiler.core.roles.base import BaseRole
from prompt_compiler.llm.factory import get_llm_provider
from prompt_compiler.llm.prompts.prompt_objects import CandidatePrompt
from prompt_compiler.utils.logging import get_logger
from prompt_compiler.utils.telemetry import telemetry
from prompt_compiler.utils.token_collector import token_collector

logger = get_logger(__name__)


@define
class DefaultPilot(IPilot, BaseRole):
    """
    Standard Pilot implementation.

    The Pilot's role is to "test fly" the new Candidate Prompt.
    """

    @property
    def role_name(self) -> str:
        return "pilot"

    async def test_candidate(self, candidate: CandidatePrompt) -> CandidatePrompt:
        """
        Execute the candidate prompt against the target model.
        """
        attributes = self._get_base_attributes()
        attributes["prompt_compiler.pilot.max_retries"] = 0  # Default implementation has no retries
        attributes["prompt_compiler.pilot.temperature"] = 0.0

        with telemetry.span(f"{self.role_name}.test_candidate", attributes=attributes):
            logger.info("Pilot testing candidate", model=candidate.model.model_name)

            provider = get_llm_provider(candidate.model.provider.provider)

            try:
                llm_response = await provider.generate(
                    system_prompt="You are a helpful assistant.",
                    user_prompt=candidate.prompt,
                    model_name=candidate.model.model_name,
                    config={"temperature": 0.0},
                )
                # Collect tokens
                token_collector.add(candidate.model.model_name, llm_response.usage)

                response = llm_response.content
                candidate.response = response
                logger.info("Pilot test complete", response_length=len(response))

            except Exception as e:
                logger.error("Pilot failed", error=str(e))
                candidate.response = f"ERROR: Execution failed. {e!s}"

            return candidate
