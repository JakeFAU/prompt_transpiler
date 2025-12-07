import json

from attrs import define

from prompt_compiler.core.exceptions import EvaluationError
from prompt_compiler.core.interfaces import IJudge
from prompt_compiler.llm.factory import get_llm_provider
from prompt_compiler.llm.prompts.prompt_objects import (
    CandidatePrompt,
    OriginalPrompt,
    ScoringAlgorithm,
)
from prompt_compiler.utils.logging import get_logger
from prompt_compiler.utils.telemetry import telemetry

logger = get_logger(__name__)


@define
class WeightedScoreAlgorithm(ScoringAlgorithm):
    """
    Standard scoring strategy.

    Weights:
    - Primary Intent: 50%
    - Tone/Voice: 30%
    - Constraints: 20% (Average of all constraint scores)
    """

    intent_weight: float = 0.5
    tone_weight: float = 0.3
    constraint_weight: float = 0.2

    def calculate_score(self, candidate: CandidatePrompt, original: OriginalPrompt) -> float:
        # If Judge hasn't run, score is 0
        if candidate.primary_intent_score is None:
            return 0.0

        score = 0.0

        # 1. Intent
        score += (candidate.primary_intent_score or 0.0) * self.intent_weight

        # 2. Tone
        score += (candidate.tone_voice_score or 0.0) * self.tone_weight

        # 3. Constraints (Average)
        if candidate.constraint_scores:
            avg_constraint = sum(candidate.constraint_scores.values()) / len(
                candidate.constraint_scores
            )
            score += avg_constraint * self.constraint_weight

        return score


@define
class LLMAdjudicator(IJudge):
    """
    Judge Role: Measures the quality of the response using an LLM.
    """

    provider_name: str = "openai"
    model_name: str = "gpt-4o"

    @telemetry.instrument(name="judge.evaluate")
    async def evaluate(self, candidate: CandidatePrompt, original: OriginalPrompt) -> float:
        """
        Runs the evaluation and returns 0.0 (legacy return).
        The component scores are updated on the Candidate object.
        """
        logger.info("Judge evaluating candidate", judge_model=self.model_name)

        provider = get_llm_provider(self.provider_name)

        system_prompt = (
            "You are a Judge. Compare the Baseline response and the Candidate response "
            "based on the Intent."
        )

        user_prompt = (
            f"Original Prompt (Context): {original.prompt}\n"
            f"Baseline Response: {original.response}\n"
            f"Candidate Response: {candidate.response}\n\n"
            "Rate the Candidate on scale 0.0 to 1.0 for: Primary Intent, Tone, Constraints.\n"
            "Also provide a short constructive hint for the Architect to improve the prompt. "
            "Do NOT leak the content of the Baseline response in the hint."
        )

        schema = {
            "type": "object",
            "properties": {
                "primary_intent_score": {"type": "number"},
                "tone_voice_score": {"type": "number"},
                "constraint_scores": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "feedback_hint": {"type": "string"},
            },
            "required": [
                "primary_intent_score",
                "tone_voice_score",
                "constraint_scores",
                "feedback_hint",
            ],
        }

        try:
            response_text = await provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.model_name,
                config={"temperature": 0.0},
                response_schema=schema,
            )

            data = json.loads(response_text)
            logger.debug("Judge scores", scores=data)

            candidate.primary_intent_score = data.get("primary_intent_score", 0.0)
            candidate.tone_voice_score = data.get("tone_voice_score", 0.0)
            candidate.constraint_scores = data.get("constraint_scores", {})
            candidate.feedback = data.get("feedback_hint", "")

            # We return 0.0 because the *Pipeline* will calculate the final score using the Strategy
            return 0.0

        except json.JSONDecodeError as e:
            logger.error("Judge received invalid JSON", error=str(e))
            raise EvaluationError("Judge returned invalid JSON") from e
        except Exception as e:
            logger.error("Judge failed", error=str(e))
            return 0.0
