"""
API route definitions for the Prompt Compiler service.

This module registers all HTTP endpoints with the Flask application, handling:
- Health checks and version information
- Compile job lifecycle (enqueue, status, result, cancel)
- Model registry operations (list, register)
- Scoring algorithm discovery

All routes use APIFlask decorators for automatic OpenAPI documentation generation.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any, cast

from apiflask import APIFlask
from flask import current_app, request

from prompt_compiler.api.errors import make_error_response
from prompt_compiler.api.schemas import (
    CompileJobEnqueueResponseSchema,
    CompileJobResultResponseSchema,
    CompileJobStatusResponseSchema,
    CompileRequestSchema,
    ModelListResponseSchema,
    ModelQuerySchema,
    ModelRegistrationSchema,
    ModelResponseSchema,
    ScoringAlgorithmListResponseSchema,
)
from prompt_compiler.core.registry import ModelRegistry
from prompt_compiler.dto.models import ModelSchema
from prompt_compiler.jobs.models import JobRecord, JobStatus
from prompt_compiler.jobs.service import JobService


def register_routes(app: APIFlask) -> None:
    @app.get("/healthz")
    @app.doc(tags=["meta"], summary="Health check", description="Simple liveness probe.")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/version")
    @app.doc(
        tags=["meta"],
        summary="API version info",
        description="Return the service version and runtime environment.",
    )
    def version() -> dict[str, str]:
        return {
            "version": _get_version(),
            "environment": current_app.config.get("PROMPT_COMPILER_ENV", "dev"),
        }

    @app.post("/v1/compile-jobs")
    @app.input(CompileRequestSchema)
    @app.output(CompileJobEnqueueResponseSchema, status_code=202)
    @app.doc(
        tags=["compile", "jobs"],
        summary="Enqueue a compile job",
        description=(
            "Submit a compile request and receive a job ID immediately. "
            "Use the status and result URLs to poll for completion."
        ),
    )
    def enqueue_compile_job(json_data: dict[str, Any]) -> dict[str, Any]:
        job_service = _get_job_service()
        job_id = job_service.enqueue_compile(json_data)
        base_url = request.host_url.rstrip("/")
        return {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "status_url": f"{base_url}/v1/compile-jobs/{job_id}",
            "result_url": f"{base_url}/v1/compile-jobs/{job_id}/result",
        }

    @app.get("/v1/compile-jobs/<job_id>")
    @app.output(CompileJobStatusResponseSchema)
    @app.doc(
        tags=["jobs"],
        summary="Get job status",
        description="Retrieve job metadata and current status.",
        responses={404: "Job not found"},
    )
    def get_job_status(job_id: str) -> tuple[dict[str, Any], int] | dict[str, Any]:
        job = _get_job_service().get_status(job_id)
        if not job:
            return make_error_response("not_found", "Job not found", {"job_id": job_id}), 404
        return _job_status_payload(job)

    @app.get("/v1/compile-jobs/<job_id>/result")
    @app.output(CompileJobResultResponseSchema, status_code=200)
    @app.doc(
        tags=["jobs", "compile"],
        summary="Get job result",
        description="Fetch the compile result if the job has completed successfully.",
        responses={
            202: "Job still running",
            404: "Job not found",
            409: "Job canceled",
            500: "Job failed",
        },
    )
    def get_job_result(job_id: str) -> tuple[dict[str, Any], int] | dict[str, Any]:
        job = _get_job_service().get_result(job_id)
        if not job:
            return make_error_response("not_found", "Job not found", {"job_id": job_id}), 404
        status = job.get("status")
        if status == JobStatus.SUCCEEDED.value:
            return {
                "job_id": job_id,
                "status": status,
                "result": _hydrate_compile_result(job.get("result")),
            }
        if status == JobStatus.FAILED.value:
            error = job.get("error")
            if isinstance(error, dict):
                code = error.get("code", "job_failed")
                message = error.get("message", "Job failed")
                details = error.get("details", {}) or {}
            else:
                code = "job_failed"
                message = "Job failed"
                details = {}
            details["job_id"] = job_id
            return make_error_response(
                code,
                message,
                details,
            ), 500
        if status == JobStatus.CANCELED.value:
            return make_error_response("job_canceled", "Job canceled", {"job_id": job_id}), 409
        return _job_status_payload(job), 202

    @app.delete("/v1/compile-jobs/<job_id>")
    @app.output(CompileJobStatusResponseSchema)
    @app.doc(
        tags=["jobs"],
        summary="Cancel a job",
        description=(
            "Best-effort cancellation. Queued jobs are canceled immediately; "
            "running jobs are marked cancel_requested."
        ),
        responses={404: "Job not found"},
    )
    def cancel_job(job_id: str) -> tuple[dict[str, Any], int] | dict[str, Any]:
        job = _get_job_service().cancel(job_id)
        if not job:
            return make_error_response("not_found", "Job not found", {"job_id": job_id}), 404
        return _job_status_payload(job)

    @app.get("/v1/models")
    @app.input(ModelQuerySchema, location="query")
    @app.output(ModelListResponseSchema)
    @app.doc(
        tags=["registry"],
        summary="List models",
        description="List registered models with optional filters.",
    )
    def list_models(query_data: dict[str, Any]) -> dict[str, Any]:
        registry = _get_registry()
        provider_filter = query_data.get("provider")
        supports_json_mode = query_data.get("supports_json_mode")

        models = list(registry._models.values())
        if provider_filter:
            models = [m for m in models if m.provider.provider == provider_filter]
        if supports_json_mode is not None:
            models = [m for m in models if m.supports_json_mode == supports_json_mode]

        models.sort(key=lambda m: m.model_name)
        # Return Model objects - the output schema will serialize them
        return {"models": models}

    @app.post("/v1/models")
    @app.input(ModelRegistrationSchema)
    @app.output(ModelResponseSchema)
    @app.doc(
        tags=["registry"],
        summary="Register a model",
        description="Register a model at runtime using a model definition payload.",
    )
    def register_model(json_data: dict[str, Any]) -> dict[str, Any]:
        registry = _get_registry()
        model = registry.register_model_from_dict(json_data)
        # Return Model object - the output schema will serialize it
        return {"model": model}

    @app.get("/v1/scoring-algorithms")
    @app.output(ScoringAlgorithmListResponseSchema)
    @app.doc(
        tags=["scoring"],
        summary="List scoring algorithms",
        description="Return supported scoring algorithms and descriptions.",
    )
    def scoring_algorithms() -> dict[str, Any]:
        return {
            "algorithms": [
                {
                    "name": "weighted",
                    "description": "Standard weighted scoring for intent, tone, and constraints.",
                },
                {
                    "name": "geometric",
                    "description": "Geometric mean scoring to penalize weak components.",
                },
                {
                    "name": "penalty",
                    "description": "Penalty-based scoring that deducts for constraint failures.",
                },
                {
                    "name": "dynamic",
                    "description": "Adaptive scoring based on prompt intent mode.",
                },
            ]
        }


def _job_status_payload(job: JobRecord) -> dict[str, Any]:
    base_url = request.host_url.rstrip("/")
    job_id = job.get("job_id")
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "stage": job.get("stage"),
        "progress": job.get("progress"),
        "cancel_requested": job.get("cancel_requested", False),
        "status_url": f"{base_url}/v1/compile-jobs/{job_id}",
        "result_url": f"{base_url}/v1/compile-jobs/{job_id}/result",
        "error": job.get("error"),
    }


def _get_job_service() -> JobService:
    return cast(JobService, current_app.extensions["job_service"])


def _get_registry() -> ModelRegistry:
    return cast(ModelRegistry, current_app.extensions["model_registry"])


def _get_version() -> str:
    try:
        return get_version("prompt-complier")
    except PackageNotFoundError:
        version_file = Path(__file__).parent.parent.parent.parent / "VERSION.txt"
        if version_file.exists():
            return version_file.read_text().strip()
        return "0.0.0"


def _hydrate_compile_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return result
    models = result.get("models")
    if not isinstance(models, dict):
        return result
    model_schema = ModelSchema()
    hydrated_models = dict(models)
    source_model = models.get("source_model")
    target_model = models.get("target_model")
    if isinstance(source_model, dict):
        hydrated_models["source_model"] = model_schema.load(source_model)
    if isinstance(target_model, dict):
        hydrated_models["target_model"] = model_schema.load(target_model)
    hydrated_result = dict(result)
    hydrated_result["models"] = hydrated_models
    return hydrated_result
