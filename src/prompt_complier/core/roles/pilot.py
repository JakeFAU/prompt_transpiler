from attrs import define

from prompt_complier.core.interfaces import IPilot
from prompt_complier.core.roles.historian import get_provider_adapter
from prompt_complier.llm.prompts.prompt_objects import CandidatePrompt
from prompt_complier.utils.logging import get_logger
from prompt_complier.utils.telemetry import telemetry

logger = get_logger(__name__)


@define
class DefaultPilot(IPilot):
    """
    Standard Pilot implementation.

    The Pilot's role is to "test fly" the new Candidate Prompt. It executes the prompt
    against the target model to see what happens.

    Limitation:
        Currently, this uses a generic input ("You are a helpful assistant" or similar dummy)
        if the IR does not provide specific test cases. Future versions should ingest
        a test dataset to run the prompt against.
    """

    @telemetry.instrument(name="pilot.test_candidate")
    async def test_candidate(self, candidate: CandidatePrompt) -> CandidatePrompt:
        """
        Execute the candidate prompt against the target model.

        Args:
            candidate: The optimized prompt object waiting to be tested.

        Returns:
            CandidatePrompt: The same object, mutated to include the 'response' string.
                             If execution fails, the response will contain the error message.
        """
        logger.info("Pilot testing candidate", model=candidate.model.model_name)

        provider = get_provider_adapter(candidate.model.provider.provider)

        try:
            # Pilot executes the prompt
            # We treat the candidate prompt as the 'user_prompt' or 'system_prompt'
            # depending on how we want to test.
            # Here we assume candidate.prompt IS the system prompt,
            # and we give a generic user input to see how it behaves,
            # OR we assume the user wants to see it run against a "test input".
            # For now, sticking to "You are a helpful assistant" baseline approach
            # implies we might need test cases.
            # Since we don't have test cases in the DTO, we use a generic placeholder.

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
            # We record the error in the response rather than crashing the pipeline
            # so the Judge can see it failed.
            candidate.response = f"ERROR: Execution failed. {e!s}"

        return candidate
