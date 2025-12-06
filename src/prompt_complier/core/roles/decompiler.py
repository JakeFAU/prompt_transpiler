import json

from attrs import define

from prompt_complier.core.exceptions import DecompilationError
from prompt_complier.core.interfaces import IDecompiler
from prompt_complier.dto.models import (
    IntermediateRepresentation,
    IntermediateRepresentationData,
    IntermediateRepresentationMeta,
    IntermediateRepresentationSpec,
    Model,
)
from prompt_complier.llm.gemini import GeminiAdapter
from prompt_complier.llm.prompts.prompt_objects import OriginalPrompt
from prompt_complier.utils.logging import get_logger
from prompt_complier.utils.telemetry import telemetry

logger = get_logger(__name__)


@define
class GeminiDecompiler(IDecompiler):
    """
    Decompiler implementation using Gemini 1.5 Pro.

    The Decompiler's role is to analyze an existing "Original Prompt" and
    reverse-engineer it into a structured, model-agnostic Intermediate Representation (IR).
    This IR serves as the blueprint for the Architect to rebuild the prompt.
    """

    model_name: str = "gemini-1.5-pro"

    @telemetry.instrument(name="decompiler.decompile")
    async def decompile(
        self, original_prompt: OriginalPrompt, target_model: Model
    ) -> IntermediateRepresentation:
        """
        Decompile a raw prompt into a structured specification.

        Args:
            original_prompt: The source prompt object containing the raw text.
            target_model: The model we eventually want to target (metadata only at this stage).

        Returns:
            IntermediateRepresentation: The structured intent, constraints, and data extracted
                                        from the prompt.

        Raises:
            DecompilationError: If the model output is not valid JSON or fails to match the schema.
        """
        logger.info("Decompiler starting analysis")

        provider = GeminiAdapter()

        system_prompt = (
            "You are an expert Prompt Engineer. "
            "Your task is to analyze the given prompt and reverse-engineer it into "
            "a structured specification (IR)."
        )

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
            logger.debug("Decompiler IR extracted", data=data)

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

            return IntermediateRepresentation(meta=meta, spec=spec, data=ir_data)

        except json.JSONDecodeError as e:
            logger.error("Decompiler received invalid JSON", error=str(e))
            raise DecompilationError("Decompiler output was not valid JSON") from e
        except Exception as e:
            logger.error("Decompiler failed", error=str(e))
            raise DecompilationError("Decompiler failed during generation") from e
