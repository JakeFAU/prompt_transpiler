"""
Marshmallow schemas for API request/response validation and serialization.

This module defines all the schemas used by the Prompt Compiler API endpoints.
Schemas are organized by domain:
- Compile request/response schemas
- Job status and result schemas
- Model registration and listing schemas
- Error response schemas
- Scoring algorithm schemas

All schemas include example metadata for OpenAPI documentation.
"""

from typing import Any, ClassVar

from apiflask import Schema
from marshmallow import fields, validate

from prompt_compiler.dto.models import ModelSchema


class RoleOverridesSchema(Schema):
    architect_provider = fields.Str(metadata={"example": "openai"})
    architect_model = fields.Str(metadata={"example": "gpt-4o"})
    decompiler_provider = fields.Str(metadata={"example": "gemini"})
    decompiler_model = fields.Str(metadata={"example": "gemini-2.5-pro"})
    diff_provider = fields.Str(metadata={"example": "openai"})
    diff_model = fields.Str(metadata={"example": "gpt-4o"})
    judge_provider = fields.Str(metadata={"example": "openai"})
    judge_model = fields.Str(metadata={"example": "gpt-4o"})


class CompileRequestSchema(Schema):
    raw_prompt = fields.Str(required=True, metadata={"example": "Summarize this text."})
    source_model = fields.Str(required=True, metadata={"example": "gpt-4o-mini"})
    target_model = fields.Str(required=True, metadata={"example": "gemini-2.5-flash"})
    max_retries = fields.Int(metadata={"example": 3})
    score_threshold = fields.Float(metadata={"example": 0.85})
    scoring_algo = fields.Str(
        validate=validate.OneOf(["weighted", "geometric", "penalty", "dynamic"]),
        metadata={"example": "weighted"},
    )
    role_overrides = fields.Dict(keys=fields.Str(), values=fields.Raw())

    class Meta:
        example: ClassVar[dict[str, Any]] = {
            "raw_prompt": "Extract stock tickers and return JSON.",
            "source_model": "gpt-4o-mini",
            "target_model": "gemini-2.5-flash",
            "max_retries": 2,
            "score_threshold": 0.85,
            "scoring_algo": "weighted",
            "role_overrides": {
                "architect_provider": "openai",
                "architect_model": "gpt-4o",
            },
        }


class CompileScoresSchema(Schema):
    primary_intent_score = fields.Float(allow_none=True, metadata={"example": 0.92})
    tone_voice_score = fields.Float(allow_none=True, metadata={"example": 0.88})
    domain_context_score = fields.Float(allow_none=True, metadata={"example": 0.86})
    constraint_scores = fields.Dict(keys=fields.Str(), values=fields.Float(), allow_none=True)
    final_score = fields.Float(metadata={"example": 0.9})


class CompileModelsSchema(Schema):
    source_model = fields.Nested(ModelSchema)
    target_model = fields.Nested(ModelSchema)


class CompileRunMetadataSchema(Schema):
    max_retries = fields.Int(metadata={"example": 3})
    score_threshold = fields.Float(metadata={"example": 0.85})
    scoring_algo = fields.Str(metadata={"example": "weighted"})
    source_provider = fields.Str(metadata={"example": "openai"})
    target_provider = fields.Str(metadata={"example": "gemini"})
    role_overrides = fields.Dict(keys=fields.Str(), values=fields.Raw())
    role_settings = fields.Dict(keys=fields.Str(), values=fields.Raw())
    requested = fields.Dict(keys=fields.Str(), values=fields.Raw())


class CompileResponseSchema(Schema):
    candidate_prompt = fields.Str(metadata={"example": "You are an expert..."})
    feedback = fields.Str(allow_none=True)
    diff_summary = fields.Str(allow_none=True)
    scores = fields.Nested(CompileScoresSchema)
    models = fields.Nested(CompileModelsSchema)
    run_metadata = fields.Nested(CompileRunMetadataSchema)


class CompileJobEnqueueResponseSchema(Schema):
    job_id = fields.Str(metadata={"example": "4f2b1b2d0a1b4a4c8ccf9d0f7a3e4d1a"})
    status = fields.Str(metadata={"example": "queued"})
    status_url = fields.Str(metadata={"example": "http://localhost:8080/v1/compile-jobs/4f2b1"})
    result_url = fields.Str(
        metadata={"example": "http://localhost:8080/v1/compile-jobs/4f2b1/result"}
    )

    class Meta:
        example: ClassVar[dict[str, Any]] = {
            "job_id": "4f2b1b2d0a1b4a4c8ccf9d0f7a3e4d1a",
            "status": "queued",
            "status_url": "http://localhost:8080/v1/compile-jobs/4f2b1b2d0a1b4a4c8ccf9d0f7a3e4d1a",
            "result_url": "http://localhost:8080/v1/compile-jobs/4f2b1b2d0a1b4a4c8ccf9d0f7a3e4d1a/result",
        }


class CompileJobStatusResponseSchema(Schema):
    job_id = fields.Str(metadata={"example": "4f2b1b2d0a1b4a4c8ccf9d0f7a3e4d1a"})
    status = fields.Str(metadata={"example": "running"})
    created_at = fields.Str(metadata={"example": "2025-01-01T00:00:00+00:00"})
    updated_at = fields.Str(metadata={"example": "2025-01-01T00:00:05+00:00"})
    started_at = fields.Str(allow_none=True)
    completed_at = fields.Str(allow_none=True)
    stage = fields.Str(allow_none=True)
    progress = fields.Dict(keys=fields.Str(), values=fields.Raw(), allow_none=True)
    cancel_requested = fields.Bool(metadata={"example": False})
    status_url = fields.Str()
    result_url = fields.Str()
    error = fields.Dict(keys=fields.Str(), values=fields.Raw(), allow_none=True)

    class Meta:
        example: ClassVar[dict[str, Any]] = {
            "job_id": "4f2b1b2d0a1b4a4c8ccf9d0f7a3e4d1a",
            "status": "running",
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:05+00:00",
            "started_at": "2025-01-01T00:00:05+00:00",
            "completed_at": None,
            "stage": "compile",
            "progress": {"step": "compile"},
            "cancel_requested": False,
            "status_url": "http://localhost:8080/v1/compile-jobs/4f2b1",
            "result_url": "http://localhost:8080/v1/compile-jobs/4f2b1/result",
            "error": None,
        }


class CompileJobResultResponseSchema(Schema):
    job_id = fields.Str()
    status = fields.Str()
    result = fields.Nested(CompileResponseSchema)

    class Meta:
        example: ClassVar[dict[str, Any]] = {
            "job_id": "4f2b1b2d0a1b4a4c8ccf9d0f7a3e4d1a",
            "status": "succeeded",
            "result": {
                "candidate_prompt": "You are an expert...",
                "feedback": None,
                "diff_summary": None,
                "scores": {
                    "primary_intent_score": 0.92,
                    "tone_voice_score": 0.88,
                    "domain_context_score": None,
                    "constraint_scores": {"json_format": 0.95},
                    "final_score": 0.9,
                },
                "models": {
                    "source_model": {"model_name": "gpt-4o-mini"},
                    "target_model": {"model_name": "gemini-2.5-flash"},
                },
                "run_metadata": {
                    "max_retries": 2,
                    "score_threshold": 0.85,
                    "scoring_algo": "weighted",
                    "source_provider": "openai",
                    "target_provider": "gemini",
                },
            },
        }


class ErrorDetailSchema(Schema):
    code = fields.Str(metadata={"example": "validation_error"})
    message = fields.Str(metadata={"example": "Invalid request payload"})
    details = fields.Dict(keys=fields.Str(), values=fields.Raw())


class ErrorResponseSchema(Schema):
    error = fields.Nested(ErrorDetailSchema)

    class Meta:
        example: ClassVar[dict[str, Any]] = {
            "error": {
                "code": "validation_error",
                "message": "Invalid request payload",
                "details": {"raw_prompt": ["Missing data for required field."]},
            }
        }


class ModelListResponseSchema(Schema):
    models = fields.List(fields.Nested(ModelSchema))

    class Meta:
        example: ClassVar[dict[str, Any]] = {
            "models": [
                {
                    "model_name": "gpt-4o-mini",
                    "supports_json_mode": True,
                    "context_window_size": 64000,
                }
            ]
        }


class ModelResponseSchema(Schema):
    model = fields.Nested(ModelSchema)

    class Meta:
        example: ClassVar[dict[str, Any]] = {
            "model": {
                "model_name": "gpt-4o-mini",
                "supports_json_mode": True,
                "context_window_size": 64000,
            }
        }


class ModelProviderSchema(Schema):
    provider = fields.Str(required=True, metadata={"example": "openai"})
    provider_type = fields.Str(required=True, metadata={"example": "api"})
    metadata = fields.Dict(keys=fields.Str(), values=fields.Raw())


class ModelRegistrationSchema(Schema):
    provider = fields.Nested(ModelProviderSchema, required=True)
    model_name = fields.Str(required=True)
    supports_system_messages = fields.Bool(required=True)
    context_window_size = fields.Int(required=True)
    prompt_style = fields.Str(required=True)
    supports_json_mode = fields.Bool(required=True)
    prompting_tips = fields.Str(required=True)
    metadata = fields.Dict(keys=fields.Str(), values=fields.Raw())


class ModelQuerySchema(Schema):
    provider = fields.Str(metadata={"example": "openai"})
    supports_json_mode = fields.Bool(metadata={"example": True})


class ScoringAlgorithmSchema(Schema):
    name = fields.Str(metadata={"example": "weighted"})
    description = fields.Str(metadata={"example": "Standard weighted scoring."})


class ScoringAlgorithmListResponseSchema(Schema):
    algorithms = fields.List(fields.Nested(ScoringAlgorithmSchema))

    class Meta:
        example: ClassVar[dict[str, Any]] = {
            "algorithms": [
                {"name": "weighted", "description": "Standard weighted scoring."},
                {"name": "geometric", "description": "Geometric mean scoring."},
            ]
        }
