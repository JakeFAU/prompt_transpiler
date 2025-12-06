from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import marshmallow as ma
from attrs import define, field, validators

from prompt_complier.dto.models import Model, ModelSchema


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
    """Marshmallow schema for serializing/deserializing OriginalPrompt."""

    prompt = ma.fields.Str(required=True)
    model = ma.fields.Nested(ModelSchema, required=True)
    response_format = ma.fields.Dict(
        keys=ma.fields.Str(), values=ma.fields.Raw(), required=False, allow_none=True
    )
    response = ma.fields.Str(required=False, allow_none=True)

    @ma.post_load
    def make_original_prompt(self, data: dict[str, Any], **kwargs: Any) -> OriginalPrompt:
        return OriginalPrompt(**data)


@define(kw_only=True)
class TranspiledPrompt:
    """Holds the transpiled prompt ready for LLM consumption."""

    prompt: str = field(validator=validators.instance_of(str))
    model: Model = field(validator=validators.instance_of(Model))
    response_format: dict[str, Any] | None = field(
        default=None, validator=validators.optional(validators.instance_of(dict))
    )
    response: str | None = field(
        default=None, validator=validators.optional(validators.instance_of(str))
    )


class TranspiledPromptSchema(ma.Schema):
    """Marshmallow schema for serializing/deserializing TranspiledPrompt."""

    prompt = ma.fields.Str(required=True)
    model = ma.fields.Nested(ModelSchema, required=True)
    response_format = ma.fields.Dict(
        keys=ma.fields.Str(), values=ma.fields.Raw(), required=False, allow_none=True
    )
    response = ma.fields.Str(required=False, allow_none=True)

    @ma.post_load
    def make_transpiled_prompt(self, data: dict[str, Any], **kwargs: Any) -> TranspiledPrompt:
        return TranspiledPrompt(**data)


# An algo is a Callable that takes a CandidatePrompt and OriginalPrompt and returns a float score
class ScoringAlgorithm(ABC):
    """Abstract base class for scoring algorithms."""

    @abstractmethod
    def calculate_score(self, candidate: "CandidatePrompt", original: OriginalPrompt) -> float:
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

    _score_algorithm: Callable[["CandidatePrompt", OriginalPrompt], float]
    _total_score: float | None

    @lru_cache(maxsize=4)  # noqa: B019
    def total_score(
        self,
        algo: Callable[["CandidatePrompt", OriginalPrompt], float],
        original_prompt: OriginalPrompt,
    ) -> float:
        """Calculate total score using the provided scoring algorithm."""
        if self._total_score is not None:
            if self._score_algorithm == algo:
                return self._total_score
        self._total_score = algo(self, original_prompt)
        self._score_algorithm = algo
        return self._total_score
