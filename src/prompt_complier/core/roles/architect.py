import json

from attrs import define

from prompt_complier.core.exceptions import ArchitectureError
from prompt_complier.core.interfaces import IArchitect
from prompt_complier.dto.models import IntermediateRepresentation, Model
from prompt_complier.llm.openai import OpenAIAdapter
from prompt_complier.llm.prompts.prompt_objects import CandidatePrompt
from prompt_complier.utils.logging import get_logger
from prompt_complier.utils.telemetry import telemetry

logger = get_logger(__name__)


@define
class GPTArchitect(IArchitect):
    """
    Architect implementation using GPT-4.

    The Architect's role is to take an Intermediate Representation (IR) of a prompt
    and synthesize a new Candidate Prompt optimized for a specific target model.
    It acts as the creative engine, applying prompt engineering best practices
    tailored to the target model's known preferences (e.g., XML tags for Claude,
    specific system instructions for Llama).
    """

    model_name: str = "gpt-4-turbo"

    @telemetry.instrument(name="architect.design_prompt")
    async def design_prompt(
        self,
        ir: IntermediateRepresentation,
        target_model: Model,
        feedback: str | None = None,
    ) -> CandidatePrompt:
        """
        Generate a new prompt candidate based on the IR and optional feedback.

        Args:
            ir: The Intermediate Representation specification of the prompt's intent.
            target_model: The model the new prompt is being designed for.
            feedback: Optional feedback from a previous iteration (e.g., from the Judge)
                      to guide the optimization.

        Returns:
            CandidatePrompt: A wrapper containing the generated prompt text and model metadata.

        Raises:
            ArchitectureError: If the LLM provider fails to generate a response.
        """
        logger.info("Architect designing prompt", target_model=target_model.model_name)

        provider = OpenAIAdapter()

        spec_text = (
            f"Primary Intent: {ir.spec.primary_intent}\n"
            f"Tone/Voice: {ir.spec.tone_voice}\n"
            f"Domain: {ir.spec.domain_context}\n"
            f"Constraints: {json.dumps(ir.spec.constraints)}\n"
            f"Input Format: {ir.spec.input_format}\n"
            f"Output Schema: {ir.spec.output_schema}\n"
        )

        examples_text = ""
        if ir.data.few_shot_examples:
            examples_text = "Few-Shot Examples:\n" + "\n".join(
                [
                    f"Input: {ex['input']}\nOutput: {ex['output']}"
                    for ex in ir.data.few_shot_examples
                ]
            )

        feedback_text = ""
        if feedback:
            feedback_text = (
                f"\n\nCRITICAL FEEDBACK FROM PREVIOUS ITERATION:\n{feedback}\n"
                "Address this feedback in your new design."
            )

        system_prompt = (
            "You are a Prompt Architect. Your goal is to write a highly optimized system "
            f"prompt for the model '{target_model.model_name}'.\n"
            f"Model Prompting Tips: {target_model.prompting_tips}\n"
            f"Target Prompt Style: {target_model.prompt_style.value}\n"
            "Do NOT look at the original prompt (Clean Room). Use ONLY the provided specification."
        )

        user_prompt = (
            f"Specification:\n{spec_text}\n{examples_text}{feedback_text}\n\n"
            "Write the optimized prompt:"
        )

        try:
            response_text = await provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.model_name,
                config={"temperature": 0.7},
            )
            logger.debug("Architect generated prompt", length=len(response_text))

            return CandidatePrompt(prompt=response_text, model=target_model)
        except Exception as e:
            logger.error("Architect failed", error=str(e))
            raise ArchitectureError("Architect failed to generate prompt") from e
