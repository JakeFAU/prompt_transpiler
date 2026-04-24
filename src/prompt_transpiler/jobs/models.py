"""Typed models for transpile job status and payloads."""

from enum import StrEnum
from typing import Any, TypedDict


class JobStatus(StrEnum):
    """Lifecycle states for a transpile job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


COMPLETED_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELED.value,
    }
)


class JobError(TypedDict, total=False):
    """Structured error details for failed jobs."""

    code: str
    message: str
    details: dict[str, Any]


class JobRecord(TypedDict, total=False):
    """Storage record for a transpile job."""

    job_id: str
    status: str
    request: dict[str, Any]
    result: dict[str, Any] | None
    error: JobError | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    stage: str | None
    progress: dict[str, Any] | None
    cancel_requested: bool
    worker_id: str | None
