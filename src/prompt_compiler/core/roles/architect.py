import json

from attrs import define

from prompt_compiler.core.exceptions import ArchitectureError
from prompt_compiler.core.interfaces import IArchitect
from prompt_compiler.dto.models import IntermediateRepresentation, Model
from prompt_compiler.llm.factory import get_llm_provider
from prompt_compiler.llm.prompts.prompt_objects import CandidatePrompt
from prompt_compiler.utils.logging import get_logger
from prompt_compiler.utils.telemetry import telemetry

logger = get_logger(__name__)


@define
class GPTArchitect(IArchitect):
    """
    Architect implementation using an LLM.

    The Architect's role is to take an Intermediate Representation (IR) of a prompt
    and synthesize a new Candidate Prompt optimized for a specific target model.
    """

    provider_name: str = "openai"
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
        """
        logger.info(
            "Architect designing prompt",
            target_model=target_model.model_name,
            architect_model=self.model_name,
        )

        provider = get_llm_provider(self.provider_name)

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
