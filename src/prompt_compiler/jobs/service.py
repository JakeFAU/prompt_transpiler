"""
Job service layer for compile job orchestration.

This module provides the business logic for managing compile jobs, including:
- Job enqueueing and status management
- Job execution via the PromptCompilerPipeline
- Role override resolution from request parameters
- Response building with proper model serialization

The JobService acts as the interface between the API layer and the underlying
job store, abstracting away storage implementation details.
"""

from collections.abc import Callable
from typing import Any

from prompt_compiler.config import settings
from prompt_compiler.core.pipeline import PromptCompilerPipeline
from prompt_compiler.core.registry import ModelRegistry
from prompt_compiler.core.roles.architect import GPTArchitect
from prompt_compiler.core.roles.decompiler import GeminiDecompiler
from prompt_compiler.core.roles.diff import SemanticDiffAgent
from prompt_compiler.core.scoring import LLMAdjudicator, get_scoring_algorithm
from prompt_compiler.dto.models import ModelSchema
from prompt_compiler.jobs.models import JobError, JobRecord, JobStatus
from prompt_compiler.jobs.store import JobStore
from prompt_compiler.jobs.util import run_coroutine_sync, utc_now_iso
from prompt_compiler.llm.prompts.prompt_objects import CandidatePrompt, OriginalPrompt
from prompt_compiler.utils.logging import get_logger

logger = get_logger(__name__)

CompilerFunc = Callable[[JobRecord, ModelRegistry], dict[str, Any]]


class JobService:
    def __init__(
        self,
        store: JobStore,
        registry: ModelRegistry,
        compiler: CompilerFunc | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self._compiler = compiler or run_compile_job

    def enqueue_compile(self, request: dict[str, Any]) -> str:
        return self.store.create_job(request)

    def get_status(self, job_id: str) -> JobRecord | None:
        return self.store.get_job(job_id)

    def get_result(self, job_id: str) -> JobRecord | None:
        return self.store.get_job(job_id)

    def cancel(self, job_id: str) -> JobRecord | None:
        job = self.store.get_job(job_id)
        if not job:
            return None
        if job.get("status") == JobStatus.QUEUED.value:
            self.store.update_job(
                job_id,
                status=JobStatus.CANCELED.value,
                completed_at=utc_now_iso(),
                stage="canceled",
            )
            return self.store.get_job(job_id)
        if job.get("status") == JobStatus.RUNNING.value:
            self.store.update_job(job_id, cancel_requested=True)
            return self.store.get_job(job_id)
        return job

    def claim_next(self, worker_id: str) -> JobRecord | None:
        return self.store.claim_next_job(worker_id)

    def run_job(self, job: JobRecord) -> None:
        job_id = job["job_id"]
        if job.get("cancel_requested"):
            self.store.update_job(
                job_id,
                status=JobStatus.CANCELED.value,
                completed_at=utc_now_iso(),
                stage="canceled",
            )
            return
        self.store.update_job(job_id, stage="compile")
        try:
            result = self._compiler(job, self.registry)
            self.store.complete_job(job_id, result)
        except Exception as exc:
            logger.error("Compile job failed", job_id=job_id, error=str(exc))
            error: JobError = {
                "code": "compile_failed",
                "message": "Compilation failed",
                "details": {"error": str(exc)},
            }
            self.store.fail_job(job_id, error)


def run_compile_job(job: JobRecord, registry: ModelRegistry) -> dict[str, Any]:
    request = job.get("request") or {}
    raw_prompt = request.get("raw_prompt", "")
    source_model = request.get("source_model", "")
    target_model = request.get("target_model", "")

    source_provider, target_provider = _resolve_providers(registry, source_model, target_model)

    max_retries = request.get("max_retries")
    score_threshold = request.get("score_threshold")
    scoring_algo = request.get("scoring_algo")

    effective_max_retries = (
        max_retries if max_retries is not None else settings.COMPILER.MAX_RETRIES
    )
    effective_score_threshold = (
        score_threshold if score_threshold is not None else settings.COMPILER.SCORE_THRESHOLD
    )

    algo_name = scoring_algo or settings.get("compiler.scoring_algorithm", "weighted")
    scoring_algorithm = get_scoring_algorithm(algo_name)

    pipeline_kwargs: dict[str, Any] = {
        "scoring_algorithm": scoring_algorithm,
        "score_threshold": effective_score_threshold,
        "max_retries": effective_max_retries,
    }

    role_overrides = request.get("role_overrides") or {}
    pipeline_kwargs.update(_build_role_overrides(role_overrides))

    pipeline = PromptCompilerPipeline(**pipeline_kwargs)

    candidate: CandidatePrompt = run_coroutine_sync(
        pipeline.run(
            raw_prompt=raw_prompt,
            source_model=source_model,
            target_model=target_model,
            source_provider=source_provider,
            target_provider=target_provider,
            max_retries=max_retries,
        )
    )

    final_score = getattr(candidate, "_cached_score", None)
    if final_score is None:
        source_model_obj = registry.get_model(source_model, source_provider)
        original = OriginalPrompt(prompt=raw_prompt, model=source_model_obj)
        final_score = candidate.total_score(scoring_algorithm, original)

    return _build_compile_response(
        {
            "candidate": candidate,
            "registry": registry,
            "request": request,
            "source_model": source_model,
            "target_model": target_model,
            "source_provider": source_provider,
            "target_provider": target_provider,
            "scoring_algo": algo_name,
            "max_retries": effective_max_retries,
            "score_threshold": effective_score_threshold,
            "final_score": final_score,
            "role_overrides": role_overrides,
            "role_settings": _effective_role_settings(role_overrides),
        }
    )


def _resolve_providers(
    registry: ModelRegistry, source_model: str, target_model: str
) -> tuple[str, str]:
    source = registry.get_model(source_model)
    target = registry.get_model(target_model)
    return source.provider.provider, target.provider.provider


def _build_role_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}

    architect_provider = overrides.get("architect_provider") or settings.roles.architect.provider
    architect_model = overrides.get("architect_model") or settings.roles.architect.model
    if overrides.get("architect_provider") or overrides.get("architect_model"):
        resolved["architect"] = GPTArchitect(
            provider_name=architect_provider,
            model_name=architect_model,
        )

    decompiler_provider = overrides.get("decompiler_provider") or settings.roles.decompiler.provider
    decompiler_model = overrides.get("decompiler_model") or settings.roles.decompiler.model
    if overrides.get("decompiler_provider") or overrides.get("decompiler_model"):
        resolved["decompiler"] = GeminiDecompiler(
            provider_name=decompiler_provider,
            model_name=decompiler_model,
        )

    diff_provider = overrides.get("diff_provider") or settings.roles.diff.provider
    diff_model = overrides.get("diff_model") or settings.roles.diff.model
    if overrides.get("diff_provider") or overrides.get("diff_model"):
        resolved["diff_agent"] = SemanticDiffAgent(
            provider_name=diff_provider,
            model_name=diff_model,
        )

    judge_provider = overrides.get("judge_provider") or settings.roles.judge.provider
    judge_model = overrides.get("judge_model") or settings.roles.judge.model
    if overrides.get("judge_provider") or overrides.get("judge_model"):
        resolved["judge"] = LLMAdjudicator(
            provider_name=judge_provider,
            model_name=judge_model,
        )

    return resolved


def _build_compile_response(payload: dict[str, Any]) -> dict[str, Any]:
    candidate: CandidatePrompt = payload["candidate"]
    registry: ModelRegistry = payload["registry"]
    request: dict[str, Any] = payload["request"]
    source_model: str = payload["source_model"]
    target_model: str = payload["target_model"]
    source_provider: str = payload["source_provider"]
    target_provider: str = payload["target_provider"]
    scoring_algo: str = payload["scoring_algo"]
    max_retries: int = payload["max_retries"]
    score_threshold: float = payload["score_threshold"]
    final_score: float = payload["final_score"]
    role_overrides: dict[str, Any] = payload["role_overrides"]
    role_settings: dict[str, Any] = payload["role_settings"]

    model_schema = ModelSchema()
    source_model_obj = registry.get_model(source_model, source_provider)
    target_model_obj = registry.get_model(target_model, target_provider)

    return {
        "candidate_prompt": candidate.prompt,
        "feedback": candidate.feedback,
        "diff_summary": candidate.diff_summary,
        "scores": {
            "primary_intent_score": candidate.primary_intent_score,
            "tone_voice_score": candidate.tone_voice_score,
            "domain_context_score": candidate.domain_context_score,
            "constraint_scores": candidate.constraint_scores,
            "final_score": final_score,
        },
        "models": {
            "source_model": model_schema.dump(source_model_obj),
            "target_model": model_schema.dump(target_model_obj),
        },
        "run_metadata": {
            "max_retries": max_retries,
            "score_threshold": score_threshold,
            "scoring_algo": scoring_algo,
            "source_provider": source_provider,
            "target_provider": target_provider,
            "role_overrides": role_overrides,
            "role_settings": role_settings,
            "requested": {
                "max_retries": request.get("max_retries"),
                "score_threshold": request.get("score_threshold"),
                "scoring_algo": request.get("scoring_algo"),
            },
        },
    }


def _effective_role_settings(overrides: dict[str, Any]) -> dict[str, Any]:
    return {
        "architect": {
            "provider": overrides.get("architect_provider") or settings.roles.architect.provider,
            "model": overrides.get("architect_model") or settings.roles.architect.model,
        },
        "decompiler": {
            "provider": overrides.get("decompiler_provider") or settings.roles.decompiler.provider,
            "model": overrides.get("decompiler_model") or settings.roles.decompiler.model,
        },
        "diff": {
            "provider": overrides.get("diff_provider") or settings.roles.diff.provider,
            "model": overrides.get("diff_model") or settings.roles.diff.model,
        },
        "judge": {
            "provider": overrides.get("judge_provider") or settings.roles.judge.provider,
            "model": overrides.get("judge_model") or settings.roles.judge.model,
        },
    }
