import json

from attrs import define

from prompt_compiler.core.exceptions import DecompilationError
from prompt_compiler.core.interfaces import IDecompiler
from prompt_compiler.core.roles.base import BaseRole
from prompt_compiler.dto.models import (
    IntermediateRepresentation,
    IntermediateRepresentationData,
    IntermediateRepresentationMeta,
    IntermediateRepresentationSpec,
    Model,
)
from prompt_compiler.llm.factory import get_llm_provider
from prompt_compiler.llm.prompts.prompt_objects import OriginalPrompt
from prompt_compiler.utils.logging import get_logger
from prompt_compiler.utils.telemetry import telemetry

logger = get_logger(__name__)


@define
class GeminiDecompiler(IDecompiler, BaseRole):
    """
    Decompiler implementation using an LLM.

    The Decompiler's role is to analyze an existing "Original Prompt" and
    reverse-engineer it into a structured, model-agnostic Intermediate Representation (IR).
    """

    provider_name: str = "gemini"
    model_name: str = "gemini-2.5-pro"

    @property
    def role_name(self) -> str:
        return "decompiler"

    def _get_system_prompt(self) -> str:
        return """
            You are an expert LLM Decompiler. Your job is to convert raw prompts into a
            Model-Agnostic Intermediate Representation (IR).

            ### CRITICAL INSTRUCTION: "Template" vs "Payload"
            You must classify the user's request into one of two types:

            1. **TYPE A: ABSTRACT TEMPLATE**
            - User asks: "Write a prompt to summarize movies."
            - Action: Create a reusable tool.
            - Variables: Extract "movies" as `{{variable}}`.

            2. **TYPE B: CONCRETE PAYLOAD (Most Common)**
            - User asks: "Summarize 'Fight Club'."
            - Action: The user wants a result NOW for this specific entity.
            - **rule**: 'Fight Club' is NOT a variable. It is **CONTEXT**.
            - **rule**: Do NOT extract specific entities as variables if the user provided them.
            - **rule**: Embed the specific data directly into the `intent` or `context` fields.

            ### JSON Output Schema
            {
                "primary_intent": "The core goal (e.g. 'Summarize Fight Club using emojis')",
                "constraints": ["No text", "Emojis only"],
                "context": "The specific data to process (e.g. 'Movie: Fight Club')",
                "variables": [],
                "tone_voice": "Playful"
            }
            // Note: For Type B requests, the "variables" array should be kept empty.
        """

    async def decompile(
        self,
        original_prompt: OriginalPrompt,
        target_model: Model,
    ) -> IntermediateRepresentation:
        """
        Decompile a raw prompt into a structured specification.
        """
        attributes = self._get_base_attributes()
        attributes["prompt_compiler.source_model"] = original_prompt.model.model_name

        with telemetry.span(f"{self.role_name}.decompile", attributes=attributes):
            logger.info(
                "Decompiler starting analysis",
                decompiler_model=self.model_name,
            )

            provider = get_llm_provider(self.provider_name)

            system_prompt = self._get_system_prompt()

            ir_schema = {
                "type": "object",
                "properties": {
                    "primary_intent": {"type": "string"},
                    "tone_voice": {"type": "string"},
                    "domain_context": {"type": "string"},
                    "constraints": {"type": "array", "items": {"type": "string"}},
                    "input_format": {"type": "string"},
                    "output_schema": {"type": "string"},
                    "few_shot_examples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string"},
                                "output": {"type": "string"},
                            },
                            "required": ["input", "output"],
                        },
                    },
                },
                "required": [
                    "primary_intent",
                    "tone_voice",
                    "domain_context",
                    "constraints",
                    "input_format",
                    "output_schema",
                    "few_shot_examples",
                ],
            }

            try:
                response_text = await provider.generate(
                    system_prompt=system_prompt,
                    user_prompt=(
                        f"Analyze this prompt and extract the specification:\n\n"
                        f"{original_prompt.prompt}"
                    ),
                    model_name=self.model_name,
                    config={"temperature": 0.0},
                    response_schema=ir_schema,
                )

                data = json.loads(response_text)
                logger.debug("Decompiler extracted IR", data=data)

                spec = IntermediateRepresentationSpec(
                    primary_intent=data["primary_intent"],
                    tone_voice=data["tone_voice"],
                    domain_context=data["domain_context"],
                    constraints=data["constraints"],
                    input_format=data["input_format"],
                    output_schema=data["output_schema"],
                )

                ir_data = IntermediateRepresentationData(
                    few_shot_examples=data.get("few_shot_examples", [])
                )

                meta = IntermediateRepresentationMeta(
                    source_model=original_prompt.model, target_model=target_model
                )

                # Metric: Success
                counter = telemetry.get_counter(
                    "decompiler.intent_extraction_success",
                    description="Count of successful IR extractions",
                )
                counter.add(1, attributes)

                return IntermediateRepresentation(meta=meta, spec=spec, data=ir_data)

            except json.JSONDecodeError as e:
                logger.error("Decompiler received invalid JSON", error=str(e))
                raise DecompilationError("Decompiler output was not valid JSON") from e
            except Exception as e:
                logger.error("Decompiler failed", error=str(e))
                raise DecompilationError("Decompiler failed during generation") from e
