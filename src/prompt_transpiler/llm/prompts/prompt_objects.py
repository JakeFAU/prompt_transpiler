"""Prompt data objects and scoring strategy interfaces."""

from abc import ABC, abstractmethod
from typing import Any

import marshmallow as ma
from attrs import define, field, validators

from prompt_transpiler.dto.models import (
    Model,
    ModelSchema,
    PromptPayload,
    PromptPayloadSchema,
)


@define(kw_only=True)
class OriginalPrompt:
    """Holds the original user prompt and optional intent."""

    payload: PromptPayload = field(validator=validators.instance_of(PromptPayload))
    model: Model = field(validator=validators.instance_of(Model))
    response: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )

    @property
    def prompt(self) -> str:
        """Returns the full text of the payload for backward compatibility."""
        return self.payload.full_text

    @property
    def response_format(self) -> dict[str, Any] | None:
        """Returns the response format from the payload for backward compatibility."""
        return self.payload.response_format


class OriginalPromptSchema(ma.Schema):
    """Marshmallow schema for serializing and deserializing OriginalPrompt."""

    payload = ma.fields.Nested(PromptPayloadSchema, required=True)
    model = ma.fields.Nested(ModelSchema, required=True)
    response = ma.fields.Str(required=False, allow_none=True)

    @ma.post_load
    def make_original_prompt(self, data: dict[str, Any], **kwargs: Any) -> OriginalPrompt:
        return OriginalPrompt(**data)


class ScoringAlgorithm(ABC):
    """
    Abstract Strategy for calculating the final score.
    This allows injecting different weighting logic (e.g. prioritizing Tone vs Intent).
    """

    @abstractmethod
    def calculate_score(self, candidate: "CandidatePrompt", original: OriginalPrompt) -> float:
        """Calculates a final float score based on the candidate's component scores."""
        pass


@define(kw_only=True)
class CompilationAttempt:
    """Machine-readable record of a single optimization attempt."""

    attempt: int = field(validator=validators.instance_of(int))
    final_score: float = field(validator=validators.instance_of(float))
    primary_intent_score: float | None = field(
        default=None, validator=validators.optional(validators.instance_of(float))
    )
    tone_voice_score: float | None = field(
        default=None, validator=validators.optional(validators.instance_of(float))
    )
    constraint_scores: dict[str, float] | None = field(
        default=None, validator=validators.optional(validators.instance_of(dict))
    )
    primary_intent_verdict: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    tone_voice_verdict: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    constraint_verdicts: dict[str, str] | None = field(
        default=None, validator=validators.optional(validators.instance_of(dict))
    )
    primary_intent_confidence: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    tone_voice_confidence: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    constraint_confidences: dict[str, str] | None = field(
        default=None, validator=validators.optional(validators.instance_of(dict))
    )
    feedback: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    accepted: bool = field(default=False, validator=validators.instance_of(bool))
    new_best: bool = field(default=False, validator=validators.instance_of(bool))


@define(kw_only=True)
class CandidatePrompt:
    """Holds a candidate prompt generated during transpilation."""

    payload: PromptPayload = field(validator=validators.instance_of(PromptPayload))
    model: Model = field(validator=validators.instance_of(Model))
    response: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )

    @property
    def prompt(self) -> str:
        """Returns the full text of the payload for backward compatibility."""
        return self.payload.full_text

    @property
    def response_format(self) -> dict[str, Any] | None:
        """Returns the response format from the payload for backward compatibility."""
        return self.payload.response_format

    # Component Scores (populated by Judge)
    primary_intent_score: float | None = field(
        default=None, validator=validators.optional(validators.instance_of(float))
    )
    tone_voice_score: float | None = field(
        default=None, validator=validators.optional(validators.instance_of(float))
    )
    domain_context_score: float | None = field(
        default=None, validator=validators.optional(validators.instance_of(float))
    )
    constraint_scores: dict[str, float] | None = field(
        default=None, validator=validators.optional(validators.instance_of(dict))
    )
    primary_intent_verdict: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    tone_voice_verdict: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    constraint_verdicts: dict[str, str] | None = field(
        default=None, validator=validators.optional(validators.instance_of(dict))
    )
    primary_intent_confidence: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    tone_voice_confidence: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )
    constraint_confidences: dict[str, str] | None = field(
        default=None, validator=validators.optional(validators.instance_of(dict))
    )

    # Feedback from the Judge for optimization
    feedback: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )

    # Semantic explanation of how this prompt differs from the original
    diff_summary: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )

    # Pipeline metadata
    attempt_history: list[CompilationAttempt] = field(
        factory=list, validator=validators.instance_of(list)
    )
    run_metadata: dict[str, Any] = field(factory=dict, validator=validators.instance_of(dict))

    # Internal Cache State (Not exposed in __init__)
    _cached_score: float | None = field(init=False, default=None)
    _cached_algo_id: int | None = field(init=False, default=None)

    def total_score(self, algo: ScoringAlgorithm, original: OriginalPrompt) -> float:
        """
        Calculates the total score using the provided strategy.
        Caches the result for the specific algorithm instance.
        """
        algo_id = id(algo)

        # Check cache hit
        if self._cached_score is not None and self._cached_algo_id == algo_id:
            return self._cached_score

        # Calculate
        score = algo.calculate_score(self, original)

        # Cache result
        self._cached_score = score
        self._cached_algo_id = algo_id

        return score
