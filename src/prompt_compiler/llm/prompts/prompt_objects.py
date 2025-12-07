from abc import ABC, abstractmethod
from typing import Any

import marshmallow as ma
from attrs import define, field, validators

from prompt_compiler.dto.models import Model, ModelSchema


@define(kw_only=True)
class OriginalPrompt:
    """Holds the original user prompt and optional intent."""

    prompt: str = field(validator=validators.instance_of(str))
    model: Model = field(validator=validators.instance_of(Model))
    response_format: dict[str, Any] | None = field(
        default=None, validator=validators.optional(validators.instance_of(dict))
    )
    response: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )


class OriginalPromptSchema(ma.Schema):
    """Marshmallow schema for serializing and deserializing OriginalPrompt."""

    prompt = ma.fields.Str(required=True)
    model = ma.fields.Nested(ModelSchema, required=True)
    response_format = ma.fields.Dict(
        keys=ma.fields.Str(), values=ma.fields.Raw(), required=False, allow_none=True
    )
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
class CandidatePrompt:
    """Holds a candidate prompt generated during transpilation."""

    prompt: str = field(validator=validators.instance_of(str))
    model: Model = field(validator=validators.instance_of(Model))
    response_format: dict[str, Any] | None = field(
        default=None, validator=validators.optional(validators.instance_of(dict))
    )
    response: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )

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

    # Feedback from the Judge for optimization
    feedback: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )

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
