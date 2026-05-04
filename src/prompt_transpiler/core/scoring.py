"""
Scoring algorithms and LLM-based evaluation for prompt quality.

This module implements the Judge role's scoring strategies, providing multiple
algorithms to evaluate how well a transpiled prompt matches the original intent:

- PairwisePreferenceAlgorithm: Weighted aggregation of pairwise verdicts
- WeightedScoreAlgorithm: Standard weighted average over component scores
- GeometricMeanAlgorithm: Strict mode - any weak component tanks the score
- PenaltyAlgorithm: Linter-style deductions for failures
- DynamicScoringAlgorithm: Adapts weights based on detected prompt type

The LLMAdjudicator prefers pairwise comparisons and derives numeric scores from
those verdicts for downstream algorithms.
"""

import json
import math

from attrs import define

from prompt_transpiler.core.exceptions import EvaluationError
from prompt_transpiler.core.interfaces import IJudge
from prompt_transpiler.core.roles.base import BaseRole
from prompt_transpiler.llm.factory import get_llm_provider
from prompt_transpiler.llm.prompts.prompt_objects import (
    CandidatePrompt,
    OriginalPrompt,
    ScoringAlgorithm,
)
from prompt_transpiler.utils.logging import get_logger
from prompt_transpiler.utils.telemetry import telemetry
from prompt_transpiler.utils.token_collector import token_collector

logger = get_logger(__name__)

CRIT_VAL: float = 0.9
MAJOR_VAL: float = 0.7
MINOR_VAL: float = 0.5
VERDICT_SCORE = {"candidate": 1.0, "tie": 0.5, "baseline": 0.0}
VALID_VERDICTS = frozenset(VERDICT_SCORE)
TIE_SCORE = VERDICT_SCORE["tie"]


@define
class PairwisePreferenceAlgorithm(ScoringAlgorithm):
    """
    Default strategy.

    Uses mechanically-derived scores from pairwise verdicts:
    candidate win = 1.0, tie = 0.5, baseline win = 0.0.
    """

    intent_weight: float = 0.5
    tone_weight: float = 0.3
    constraint_weight: float = 0.2

    def calculate_score(self, candidate: CandidatePrompt, original: OriginalPrompt) -> float:
        score = 0.0
        intent_score = _component_score(
            candidate.primary_intent_verdict, candidate.primary_intent_score
        )
        tone_score = _component_score(candidate.tone_voice_verdict, candidate.tone_voice_score)
        score += intent_score * self.intent_weight
        score += tone_score * self.tone_weight

        constraint_score = _constraint_average(candidate)
        if constraint_score is not None:
            score += constraint_score * self.constraint_weight

        return score


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
class GeometricMeanAlgorithm(ScoringAlgorithm):
    """
    Strict 'Compiler Mode'. If any component is weak, the whole score tanks.
    Best for code generation, JSON formatting, or strict constraints.
    """

    def calculate_score(self, candidate: CandidatePrompt, original: OriginalPrompt) -> float:
        # Use a small epsilon to prevent log(0) or total zeroing if strictness isn't desired
        epsilon = 0.01

        scores = [
            max(candidate.primary_intent_score or 0.0, epsilon),
            max(candidate.tone_voice_score or 0.0, epsilon),
        ]

        # Add individual constraint scores
        if candidate.constraint_scores:
            scores.extend([max(s, epsilon) for s in candidate.constraint_scores.values()])

        # Geometric Mean = (x1 * x2 * ... * xn)^(1/n)
        product = math.prod(scores)
        return float(pow(product, 1.0 / len(scores)))


@define
class PenaltyAlgorithm(ScoringAlgorithm):
    """
    Linter style. Starts at 100% and deducts points for failures.
    Good for ensuring compliance without demanding perfection.
    """

    MINOR_PENALTY: float = 0.1
    MAJOR_PENALTY: float = 0.5
    CRITICAL_PENALTY: float = 1.0

    def calculate_score(self, candidate: CandidatePrompt, original: OriginalPrompt) -> float:
        score = 1.0

        # 1. Intent Failure (Critical)
        if (candidate.primary_intent_score or 0) < CRIT_VAL:
            score -= self.CRITICAL_PENALTY

        # 2. Constraints (Major)
        if candidate.constraint_scores:
            for _rule, compliance in candidate.constraint_scores.items():
                if compliance < MAJOR_VAL:
                    score -= self.MAJOR_PENALTY

        # 3. Tone (Minor)
        if (candidate.tone_voice_score or 0) < MINOR_VAL:
            score -= self.MINOR_PENALTY

        return max(0.0, score)


@define
class DynamicScoringAlgorithm(ScoringAlgorithm):
    """
    Adapts based on the prompt type detected by the Decompiler.
    """

    def calculate_score(self, candidate: CandidatePrompt, original: OriginalPrompt) -> float:
        # Check metadata from the Decompiler step (you'll need to populate this in decompiler.py)

        mode = getattr(original, "mode", "balanced")

        if mode == "strict_code":
            weights = {"intent": 0.4, "tone": 0.0, "constraints": 0.6}
        elif mode == "creative":
            weights = {"intent": 0.4, "tone": 0.5, "constraints": 0.1}
        else:
            weights = {"intent": 0.5, "tone": 0.2, "constraints": 0.3}

        score = 0.0
        score += (candidate.primary_intent_score or 0) * weights["intent"]
        score += (candidate.tone_voice_score or 0) * weights["tone"]

        if candidate.constraint_scores:
            avg_constraint = sum(candidate.constraint_scores.values()) / len(
                candidate.constraint_scores
            )
            score += avg_constraint * weights["constraints"]

        return score


_ALGORITHM_CLASSES: dict[str, type[ScoringAlgorithm]] = {
    "pairwise": PairwisePreferenceAlgorithm,
    "weighted": WeightedScoreAlgorithm,
    "geometric": GeometricMeanAlgorithm,
    "penalty": PenaltyAlgorithm,
    "dynamic": DynamicScoringAlgorithm,
}


def get_scoring_algorithm(name: str) -> ScoringAlgorithm:
    """Factory to get scoring algorithm by name."""
    # Default to pairwise if unknown
    cls = _ALGORITHM_CLASSES.get(name.lower(), PairwisePreferenceAlgorithm)
    return cls()


@define
class LLMAdjudicator(IJudge, BaseRole):
    """
    Judge Role: Measures the quality of the response using an LLM.
    """

    provider_name: str = "openai"
    model_name: str = "gpt-4o"
    scoring_algorithm: ScoringAlgorithm | None = None
    score_threshold: float = 0.0

    @property
    def role_name(self) -> str:
        return "judge"

    async def evaluate(self, candidate: CandidatePrompt, original: OriginalPrompt) -> float:
        """
        Runs the evaluation and returns 0.0 (legacy return).
        The component scores are updated on the Candidate object.

        Args:
            candidate: The candidate prompt being evaluated.
            original: The original prompt serving as the baseline.

        Returns:
            0.0 as a legacy float return.

        Raises:
            EvaluationError: If the judge returns invalid JSON or fails evaluation.
        """
        attributes = self._get_base_attributes()
        if self.scoring_algorithm:
            attributes["prompt_transpiler.judge.algorithm"] = (
                self.scoring_algorithm.__class__.__name__
            )
        attributes["prompt_transpiler.judge.strictness"] = self.score_threshold

        with telemetry.span(f"{self.role_name}.evaluate", attributes=attributes):
            logger.info("Judge evaluating candidate", judge_model=self.model_name)

            provider = get_llm_provider(self.provider_name)

            system_prompt = (
                "You are a Judge. Compare the Baseline response and the Candidate response. "
                "For each category, decide whether the candidate is better, the baseline is "
                "better, or they are tied. Be conservative about ties and avoid inflated scoring."
            )

            user_prompt = (
                f"Original Prompt (Context): {original.prompt}\n"
                f"Baseline Response: {original.response}\n"
                f"Candidate Response: {candidate.response}\n\n"
                "Evaluate these categories:\n"
                "1. Primary Intent preservation\n"
                "2. Tone and voice match\n"
                "3. Constraint satisfaction\n\n"
                "Return verdicts using only baseline, candidate, or tie.\n"
                "Use confidence values weak, medium, or strong.\n"
                "For constraints, include one entry per material constraint you can identify.\n"
                "Also provide a short constructive hint for the Architect to improve the prompt. "
                "Do NOT leak the content of the Baseline response in the hint."
            )

            schema = {
                "type": "object",
                "properties": {
                    "primary_intent_verdict": {
                        "type": "string",
                        "enum": ["baseline", "candidate", "tie"],
                    },
                    "primary_intent_confidence": {
                        "type": "string",
                        "enum": ["weak", "medium", "strong"],
                    },
                    "tone_voice_verdict": {
                        "type": "string",
                        "enum": ["baseline", "candidate", "tie"],
                    },
                    "tone_voice_confidence": {
                        "type": "string",
                        "enum": ["weak", "medium", "strong"],
                    },
                    "constraint_verdicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "constraint": {"type": "string"},
                                "verdict": {
                                    "type": "string",
                                    "enum": ["baseline", "candidate", "tie"],
                                },
                                "confidence": {
                                    "type": "string",
                                    "enum": ["weak", "medium", "strong"],
                                },
                            },
                            "required": ["constraint", "verdict", "confidence"],
                        },
                    },
                    "feedback_hint": {"type": "string"},
                },
                "required": [
                    "primary_intent_verdict",
                    "primary_intent_confidence",
                    "tone_voice_verdict",
                    "tone_voice_confidence",
                    "constraint_verdicts",
                    "feedback_hint",
                ],
            }

            try:
                llm_response = await provider.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=self.model_name,
                    config={"temperature": 0.0},
                    response_schema=schema,
                )
                # Collect tokens
                token_collector.add(self.model_name, llm_response.usage)

                response_text = llm_response.content
                data = json.loads(response_text)
                logger.debug("Judge scores", scores=data)

                if "primary_intent_verdict" in data:
                    self._apply_pairwise_verdicts(candidate, data)
                else:
                    self._apply_numeric_scores(candidate, data)

                # We return 0.0 because the *Pipeline* will calculate the final score
                # using the Strategy
                return 0.0

            except json.JSONDecodeError as e:
                logger.error("Judge received invalid JSON", error=str(e))
                raise EvaluationError("Judge returned invalid JSON") from e
            except Exception as e:
                logger.error("Judge failed", error=str(e))
                return 0.0

    def _apply_pairwise_verdicts(self, candidate: CandidatePrompt, data: dict[str, object]) -> None:
        candidate.primary_intent_verdict = _normalize_verdict(data.get("primary_intent_verdict"))
        candidate.tone_voice_verdict = _normalize_verdict(data.get("tone_voice_verdict"))
        candidate.primary_intent_confidence = _normalize_confidence(
            data.get("primary_intent_confidence")
        )
        candidate.tone_voice_confidence = _normalize_confidence(data.get("tone_voice_confidence"))
        candidate.primary_intent_score = _verdict_to_score(candidate.primary_intent_verdict)
        candidate.tone_voice_score = _verdict_to_score(candidate.tone_voice_verdict)

        raw_constraints = _as_list(data.get("constraint_verdicts"))
        constraint_scores: dict[str, float] = {}
        constraint_verdicts: dict[str, str] = {}
        constraint_confidences: dict[str, str] = {}
        for item in raw_constraints:
            if not isinstance(item, dict):
                continue
            constraint = item.get("constraint")
            verdict = _normalize_verdict(item.get("verdict"))
            confidence = _normalize_confidence(item.get("confidence"))
            if constraint is None or verdict is None:
                continue
            name = str(constraint)
            constraint_verdicts[name] = verdict
            constraint_scores[name] = _verdict_to_score(verdict)
            if confidence is not None:
                constraint_confidences[name] = confidence

        candidate.constraint_verdicts = constraint_verdicts
        candidate.constraint_scores = constraint_scores
        candidate.constraint_confidences = constraint_confidences or None
        candidate.feedback = str(data.get("feedback_hint", ""))

    def _apply_numeric_scores(self, candidate: CandidatePrompt, data: dict[str, object]) -> None:
        candidate.primary_intent_score = _as_float(data.get("primary_intent_score"))
        candidate.tone_voice_score = _as_float(data.get("tone_voice_score"))
        raw_constraints = _as_list(data.get("constraint_scores"))
        constraint_scores = {}
        for item in raw_constraints:
            if not isinstance(item, dict):
                continue
            constraint = item.get("constraint")
            score = item.get("score")
            if constraint is not None and score is not None:
                constraint_scores[str(constraint)] = _as_float(score)
        candidate.constraint_scores = constraint_scores
        candidate.feedback = str(data.get("feedback_hint", ""))


def _normalize_verdict(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in VALID_VERDICTS:
        return normalized
    return None


def _normalize_confidence(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"weak", "medium", "strong"}:
        return normalized
    return None


def _verdict_to_score(verdict: str | None) -> float:
    if verdict is None:
        return TIE_SCORE
    return VERDICT_SCORE.get(verdict, TIE_SCORE)


def _component_score(verdict: str | None, fallback: float | None) -> float:
    if verdict is not None:
        return _verdict_to_score(verdict)
    return fallback or 0.0


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return []


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _constraint_average(candidate: CandidatePrompt) -> float | None:
    if candidate.constraint_verdicts:
        scores = [_verdict_to_score(verdict) for verdict in candidate.constraint_verdicts.values()]
        return sum(scores) / len(scores) if scores else None
    if candidate.constraint_scores:
        return sum(candidate.constraint_scores.values()) / len(candidate.constraint_scores)
    return None
