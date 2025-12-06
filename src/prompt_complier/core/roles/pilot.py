from attrs import define

from prompt_complier.core.interfaces import IPilot
from prompt_complier.llm.factory import get_llm_provider
from prompt_complier.llm.prompts.prompt_objects import CandidatePrompt
from prompt_complier.utils.logging import get_logger
from prompt_complier.utils.telemetry import telemetry

logger = get_logger(__name__)


@define
class DefaultPilot(IPilot):
    """
    Standard Pilot implementation.

    The Pilot's role is to "test fly" the new Candidate Prompt.
    """

    @telemetry.instrument(name="pilot.test_candidate")
    async def test_candidate(self, candidate: CandidatePrompt) -> CandidatePrompt:
        """
        Execute the candidate prompt against the target model.
        """
        logger.info("Pilot testing candidate", model=candidate.model.model_name)

        provider = get_llm_provider(candidate.model.provider.provider)

        try:
            response = await provider.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt=candidate.prompt,
                model_name=candidate.model.model_name,
                config={"temperature": 0.0},
            )
            candidate.response = response
            logger.info("Pilot test complete", response_length=len(response))

        except Exception as e:
            logger.error("Pilot failed", error=str(e))
            candidate.response = f"ERROR: Execution failed. {e!s}"

        return candidate
