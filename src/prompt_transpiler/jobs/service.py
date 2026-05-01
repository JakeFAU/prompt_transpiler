"""
Job service layer for transpile job orchestration.

This module provides the business logic for managing transpile jobs, including:
- Job enqueueing and status management
- Job execution via the PromptTranspilerPipeline
- Role override resolution from request parameters
- Response building with proper model serialization

The JobService acts as the interface between the API layer and the underlying
job store, abstracting away storage implementation details.
"""

from collections.abc import Callable
from typing import Any

from prompt_transpiler.config import settings
from prompt_transpiler.core.pipeline import PromptTranspilerPipeline
from prompt_transpiler.core.registry import ModelRegistry
from prompt_transpiler.core.roles.architect import GPTArchitect
from prompt_transpiler.core.roles.decompiler import GeminiDecompiler
from prompt_transpiler.core.roles.diff import SemanticDiffAgent
from prompt_transpiler.core.scoring import LLMAdjudicator, get_scoring_algorithm
from prompt_transpiler.dto.models import Message, PromptPayload
from prompt_transpiler.jobs.models import JobError, JobRecord, JobStatus
from prompt_transpiler.jobs.store import JobStore
from prompt_transpiler.jobs.util import run_coroutine_sync, utc_now_iso
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt, OriginalPrompt
from prompt_transpiler.reporting import build_transpile_report
from prompt_transpiler.utils.logging import get_logger
from prompt_transpiler.utils.token_collector import token_collector

logger = get_logger(__name__)

CompilerFunc = Callable[[JobRecord, ModelRegistry], dict[str, Any]]


class JobService:
    """Service layer that manages transpile job lifecycle and execution."""

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
        """Create a new transpile job and return its identifier."""
        return self.store.create_job(request)

    def get_status(self, job_id: str) -> JobRecord | None:
        """Fetch job status and metadata."""
        return self.store.get_job(job_id)

    def get_result(self, job_id: str) -> JobRecord | None:
        """Fetch job metadata for result polling."""
        return self.store.get_job(job_id)

    def cancel(self, job_id: str) -> JobRecord | None:
        """Cancel a job or mark it as cancel_requested if already running."""
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
        """Claim the next available job for the given worker."""
        return self.store.claim_next_job(worker_id)

    def run_job(self, job: JobRecord) -> None:
        """Run a claimed job and persist the result or failure."""
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
                "message": "Transpilation failed",
                "details": {"error": str(exc)},
            }
            self.store.fail_job(job_id, error)


def run_compile_job(job: JobRecord, registry: ModelRegistry) -> dict[str, Any]:
    """Execute a transpile job payload and return the API response payload."""
    usage_before = _token_usage_snapshot()
    request = job.get("request") or {}
    raw_prompt = request.get("raw_prompt", "")
    source_model = request.get("source_model", "")
    target_model = request.get("target_model", "")

    source_provider, target_provider = _resolve_providers(registry, source_model, target_model)

    max_retries = request.get("max_retries")
    score_threshold = request.get("score_threshold")
    scoring_algo = request.get("scoring_algo")

    effective_max_retries = (
        max_retries if max_retries is not None else settings.TRANSPILER.MAX_RETRIES
    )
    effective_score_threshold = (
        score_threshold if score_threshold is not None else settings.TRANSPILER.SCORE_THRESHOLD
    )

    algo_name = scoring_algo or settings.get("transpiler.scoring_algorithm", "weighted")
    scoring_algorithm = get_scoring_algorithm(algo_name)

    pipeline_kwargs: dict[str, Any] = {
        "scoring_algorithm": scoring_algorithm,
        "scoring_algorithm_name": algo_name,
        "score_threshold": effective_score_threshold,
        "max_retries": effective_max_retries,
    }

    role_overrides = request.get("role_overrides") or {}
    pipeline_kwargs.update(_build_role_overrides(role_overrides))

    pipeline = PromptTranspilerPipeline(**pipeline_kwargs)

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
    usage_after = _token_usage_snapshot()
    token_usage = _token_usage_delta(usage_before, usage_after)

    final_score = getattr(candidate, "_cached_score", None)
    if final_score is None:
        source_model_obj = registry.get_model(source_model, source_provider)
        original = OriginalPrompt(
            payload=PromptPayload(messages=[Message(role="user", content=raw_prompt)]),
            model=source_model_obj,
        )
        final_score = candidate.total_score(scoring_algorithm, original)

    candidate.run_metadata = {
        **candidate.run_metadata,
        "max_retries": effective_max_retries,
        "score_threshold": effective_score_threshold,
        "scoring_algo": algo_name,
        "source_provider": source_provider,
        "target_provider": target_provider,
        "role_overrides": role_overrides,
        "role_settings": _effective_role_settings(role_overrides),
        "requested": {
            "max_retries": request.get("max_retries"),
            "score_threshold": request.get("score_threshold"),
            "scoring_algo": request.get("scoring_algo"),
        },
    }

    return build_transpile_report(
        candidate,
        source_model=registry.get_model(source_model, source_provider),
        target_model=registry.get_model(target_model, target_provider),
        final_score=final_score,
        token_usage=token_usage,
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


_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


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


def _token_usage_snapshot() -> dict[str, dict[str, int]]:
    summary = token_collector.get_summary()
    return {
        model: {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        for model, usage in summary.items()
    }


def _token_usage_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    delta: dict[str, dict[str, int]] = {}
    for model, usage in after.items():
        prior = before.get(model, _ZERO_USAGE)

        # Performance optimization: extract deltas into local variables
        # and explicitly check for > 0 before allocating the dictionary,
        # avoiding unnecessary object allocation overhead
        p_diff = max(0, usage["prompt_tokens"] - prior["prompt_tokens"])
        c_diff = max(0, usage["completion_tokens"] - prior["completion_tokens"])
        t_diff = max(0, usage["total_tokens"] - prior["total_tokens"])

        if p_diff > 0 or c_diff > 0 or t_diff > 0:
            delta[model] = {
                "prompt_tokens": p_diff,
                "completion_tokens": c_diff,
                "total_tokens": t_diff,
            }
    return delta
