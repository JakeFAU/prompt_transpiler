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
