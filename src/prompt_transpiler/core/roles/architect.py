"""Architect role implementation for prompt synthesis."""

import json

from attrs import define

from prompt_transpiler.core.exceptions import ArchitectureError
from prompt_transpiler.core.interfaces import IArchitect
from prompt_transpiler.core.roles.base import BaseRole
from prompt_transpiler.dto.models import (
    IntermediateRepresentation,
    Message,
    Model,
    PromptPayload,
)
from prompt_transpiler.llm.factory import get_llm_provider
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt
from prompt_transpiler.utils.logging import get_logger
from prompt_transpiler.utils.telemetry import telemetry
from prompt_transpiler.utils.token_collector import token_collector

logger = get_logger(__name__)


@define
class GPTArchitect(IArchitect, BaseRole):
    """
    Architect implementation using an LLM.

    The Architect's role is to take an Intermediate Representation (IR) of a prompt
    and synthesize a new Candidate Prompt optimized for a specific target model.
    """

    provider_name: str = "openai"
    model_name: str = "gpt-4-turbo"

    @property
    def role_name(self) -> str:
        return "architect"

    async def design_prompt(
        self,
        ir: IntermediateRepresentation,
        target_model: Model,
        feedback: str | None = None,
    ) -> CandidatePrompt:
        """
        Generate a new prompt candidate based on the IR and optional feedback.
        """
        attributes = self._get_base_attributes()
        attributes["prompt_transpiler.target_model"] = target_model.model_name

        # Determine strategy
        strategy = "zero_shot"
        if ir.data.few_shot_examples:
            strategy = "few_shot"
        # We could also incorporate prompt_style if relevant
        # attributes["prompt_transpiler.strategy"] = f"{strategy}_{target_model.prompt_style.value}"
        attributes["prompt_transpiler.strategy"] = strategy

        with telemetry.span(f"{self.role_name}.design_prompt", attributes=attributes) as span:
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
                "Do NOT look at the original prompt (Clean Room). "
                "Use ONLY the provided specification."
            )

            user_prompt = (
                f"Specification:\n{spec_text}\n{examples_text}{feedback_text}\n\n"
                "Write the optimized prompt:"
            )

            try:
                llm_response = await provider.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=self.model_name,
                    config={"temperature": 0.7},
                )
                # Collect tokens
                token_collector.add(self.model_name, llm_response.usage)

                response_text = llm_response.content
                logger.debug("Architect generated prompt", length=len(response_text))

                if span:
                    span.set_attribute(
                        "gen_ai.response.completion_tokens", len(response_text.split())
                    )  # Approximate

                # 1. Routing Logic
                messages = []
                if target_model.supports_system_instructions:
                    # Instructions and Architect's optimized prompt go to System
                    system_content = (
                        f"Primary Intent: {ir.spec.primary_intent}\n"
                        f"Tone/Voice: {ir.spec.tone_voice}\n"
                        f"Constraints: {json.dumps(ir.spec.constraints)}\n\n"
                        f"{response_text}"
                    )
                    messages.append(Message(role="system", content=system_content))

                    # Context and data go to User
                    user_content = (
                        f"Domain Context: {ir.spec.domain_context}\n"
                        f"Input Format: {ir.spec.input_format}"
                    )
                    messages.append(Message(role="user", content=user_content))
                else:
                    # Merge all components into a single user message
                    flat_content = (
                        f"Primary Intent: {ir.spec.primary_intent}\n"
                        f"Tone/Voice: {ir.spec.tone_voice}\n"
                        f"Constraints: {json.dumps(ir.spec.constraints)}\n"
                        f"Domain Context: {ir.spec.domain_context}\n"
                        f"Input Format: {ir.spec.input_format}\n\n"
                        f"{response_text}"
                    )
                    messages.append(Message(role="user", content=flat_content))

                # 2. Structured Output
                response_format = None
                if target_model.supports_structured_outputs and ir.spec.output_schema:
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "output_schema",
                            "schema": ir.spec.output_schema,
                            "strict": True,
                        },
                    }

                payload = PromptPayload(messages=messages, response_format=response_format)
                return CandidatePrompt(payload=payload, model=target_model)
            except Exception as e:
                logger.error("Architect failed", error=str(e))
                raise ArchitectureError("Architect failed to generate prompt") from e
