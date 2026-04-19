"""Typed models for compile job status and payloads."""

from enum import StrEnum
from typing import Any, TypedDict


class JobStatus(StrEnum):
    """Lifecycle states for a compile job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class JobError(TypedDict, total=False):
    """Structured error details for failed jobs."""

    code: str
    message: str
    details: dict[str, Any]


class JobRecord(TypedDict, total=False):
    """Storage record for a compile job."""

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
