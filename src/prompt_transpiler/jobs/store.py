"""
Job storage backends for transpile job persistence.

This module provides multiple job store implementations:

- DuckDBJobStore: High-performance analytics-oriented storage (default)
- SQLiteJobStore: Lightweight file-based storage for simpler deployments
- MemoryJobStore: In-memory storage for testing and development

All stores implement the JobStore protocol, ensuring consistent behavior
across implementations. Jobs are stored with full metadata including
request payloads, results, errors, and lifecycle timestamps.
"""

import sqlite3
import threading
from typing import Any, Protocol, cast

try:
    import duckdb as _duckdb
except ImportError:  # pragma: no cover - optional dependency
    _duckdb = None  # type: ignore[assignment]

from prompt_transpiler.jobs.models import JobError, JobRecord, JobStatus
from prompt_transpiler.jobs.util import generate_job_id, json_dumps, json_loads, utc_now_iso


class JobStore(Protocol):
    """Protocol defining the storage interface for transpile jobs."""

    def create_job(self, request: dict[str, Any]) -> str:
        """Persist a new job request and return its identifier."""
        ...

    def get_job(self, job_id: str) -> JobRecord | None:
        """Fetch a job record by ID."""
        ...

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update fields on a job record."""
        ...

    def claim_next_job(self, worker_id: str) -> JobRecord | None:
        """Claim the next queued job for a worker."""
        ...

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        """Mark a job as succeeded and store its result."""
        ...

    def fail_job(self, job_id: str, error: JobError) -> None:
        """Mark a job as failed and store its error details."""
        ...

    def purge_expired(self, cutoff_iso: str) -> int:
        """Purge completed jobs older than the cutoff timestamp."""
        ...

    def close(self) -> None:
        """Release any underlying resources held by the store."""
        ...


class DuckDBJobStore:
    """DuckDB-backed job store optimized for analytics-friendly storage."""

    def __init__(self, db_path: str) -> None:
        if _duckdb is None:  # pragma: no cover - only for missing dependency
            raise RuntimeError("DuckDBJobStore requires the 'duckdb' package to be installed.")

        self._duckdb = _duckdb
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = _duckdb.connect(database=db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Ensure the transpile jobs table exists."""
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compile_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    stage TEXT,
                    progress_json TEXT,
                    cancel_requested BOOLEAN,
                    worker_id TEXT
                );
                """
            )

    def create_job(self, request: dict[str, Any]) -> str:
        """Create a new job record and return its identifier."""
        job_id = generate_job_id()
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO compile_jobs (
                    job_id, status, request_json, created_at, updated_at, cancel_requested
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    job_id,
                    JobStatus.QUEUED.value,
                    json_dumps(request),
                    now,
                    now,
                    False,
                ],
            )
        return job_id

    def get_job(self, job_id: str) -> JobRecord | None:
        """Fetch a job record by ID."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT job_id, status, request_json, result_json, error_json,
                       created_at, updated_at, started_at, completed_at,
                       stage, progress_json, cancel_requested, worker_id
                FROM compile_jobs WHERE job_id = ?
                """,
                [job_id],
            ).fetchone()
        if not row:
            return None
        return _row_to_record(row)

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update fields on a job record."""
        if not fields:
            return
        fields["updated_at"] = utc_now_iso()
        columns, values = _to_update_clause(fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE compile_jobs SET {columns} WHERE job_id = ?",  # nosec B608
                [*values, job_id],
            )

    def claim_next_job(self, worker_id: str) -> JobRecord | None:
        """Atomically claim the next queued job for a worker."""
        with self._lock:
            self._conn.execute("BEGIN TRANSACTION")
            row = self._conn.execute(
                """
                SELECT job_id FROM compile_jobs
                WHERE status = ?
                ORDER BY created_at
                LIMIT 1
                """,
                [JobStatus.QUEUED.value],
            ).fetchone()
            if not row:
                self._conn.execute("COMMIT")
                return None
            job_id = row[0]
            now = utc_now_iso()
            self._conn.execute(
                """
                UPDATE compile_jobs
                SET status = ?, started_at = ?, updated_at = ?, stage = ?, worker_id = ?
                WHERE job_id = ? AND status = ?
                """,
                [
                    JobStatus.RUNNING.value,
                    now,
                    now,
                    "claimed",
                    worker_id,
                    job_id,
                    JobStatus.QUEUED.value,
                ],
            )
            self._conn.execute("COMMIT")
        return self.get_job(job_id)

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        """Mark a job as succeeded and store its result."""
        self.update_job(
            job_id,
            status=JobStatus.SUCCEEDED.value,
            result_json=json_dumps(result),
            completed_at=utc_now_iso(),
            stage="completed",
        )

    def fail_job(self, job_id: str, error: JobError) -> None:
        """Mark a job as failed and store its error details."""
        self.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error_json=json_dumps(error),
            completed_at=utc_now_iso(),
            stage="failed",
        )

    def purge_expired(self, cutoff_iso: str) -> int:
        """Delete completed jobs older than the cutoff timestamp."""
        with self._lock:
            res = self._conn.execute(
                """
                DELETE FROM compile_jobs
                WHERE completed_at IS NOT NULL
                  AND completed_at < ?
                  AND status IN (?, ?, ?)
                """,
                [
                    cutoff_iso,
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELED.value,
                ],
            ).fetchone()
            return int(res[0]) if res else 0

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class SQLiteJobStore:
    """SQLite-backed job store for lightweight deployments."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Ensure the transpile jobs table exists."""
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compile_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    stage TEXT,
                    progress_json TEXT,
                    cancel_requested INTEGER,
                    worker_id TEXT
                );
                """
            )
            self._conn.commit()

    def create_job(self, request: dict[str, Any]) -> str:
        """Create a new job record and return its identifier."""
        job_id = generate_job_id()
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO compile_jobs (
                    job_id, status, request_json, created_at, updated_at, cancel_requested
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    job_id,
                    JobStatus.QUEUED.value,
                    json_dumps(request),
                    now,
                    now,
                    0,
                ],
            )
            self._conn.commit()
        return job_id

    def get_job(self, job_id: str) -> JobRecord | None:
        """Fetch a job record by ID."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT job_id, status, request_json, result_json, error_json,
                       created_at, updated_at, started_at, completed_at,
                       stage, progress_json, cancel_requested, worker_id
                FROM compile_jobs WHERE job_id = ?
                """,
                [job_id],
            ).fetchone()
        if not row:
            return None
        return _row_to_record(tuple(row))

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update fields on a job record."""
        if not fields:
            return
        fields["updated_at"] = utc_now_iso()
        columns, values = _to_update_clause(fields)
        with self._lock:
            self._conn.execute(
                f"UPDATE compile_jobs SET {columns} WHERE job_id = ?",  # nosec B608
                [*values, job_id],
            )
            self._conn.commit()

    def claim_next_job(self, worker_id: str) -> JobRecord | None:
        """Atomically claim the next queued job for a worker."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                """
                SELECT job_id FROM compile_jobs
                WHERE status = ?
                ORDER BY created_at
                LIMIT 1
                """,
                [JobStatus.QUEUED.value],
            ).fetchone()
            if not row:
                self._conn.execute("COMMIT")
                return None
            job_id = row[0]
            now = utc_now_iso()
            self._conn.execute(
                """
                UPDATE compile_jobs
                SET status = ?, started_at = ?, updated_at = ?, stage = ?, worker_id = ?
                WHERE job_id = ? AND status = ?
                """,
                [
                    JobStatus.RUNNING.value,
                    now,
                    now,
                    "claimed",
                    worker_id,
                    job_id,
                    JobStatus.QUEUED.value,
                ],
            )
            self._conn.commit()
        return self.get_job(job_id)

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        """Mark a job as succeeded and store its result."""
        self.update_job(
            job_id,
            status=JobStatus.SUCCEEDED.value,
            result_json=json_dumps(result),
            completed_at=utc_now_iso(),
            stage="completed",
        )

    def fail_job(self, job_id: str, error: JobError) -> None:
        """Mark a job as failed and store its error details."""
        self.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error_json=json_dumps(error),
            completed_at=utc_now_iso(),
            stage="failed",
        )

    def purge_expired(self, cutoff_iso: str) -> int:
        """Delete completed jobs older than the cutoff timestamp."""
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM compile_jobs
                WHERE completed_at IS NOT NULL
                  AND completed_at < ?
                  AND status IN (?, ?, ?)
                """,
                [
                    cutoff_iso,
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELED.value,
                ],
            )
            self._conn.commit()
            return int(cur.rowcount or 0)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MemoryJobStore:
    """In-memory job store for tests and local development."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._queued_job_ids: dict[str, None] = {}

    def create_job(self, request: dict[str, Any]) -> str:
        """Create a new job record and return its identifier."""
        job_id = generate_job_id()
        now = utc_now_iso()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatus.QUEUED.value,
                "request": request,
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
                "stage": None,
                "progress": None,
                "cancel_requested": False,
                "worker_id": None,
            }
            self._queued_job_ids[job_id] = None
        return job_id

    def get_job(self, job_id: str) -> JobRecord | None:
        """Fetch a job record by ID."""
        with self._lock:
            job = self._jobs.get(job_id)
            return cast(JobRecord, dict(job)) if job else None

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update fields on a job record."""
        if not fields:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(cast(JobRecord, fields))
            job["updated_at"] = utc_now_iso()

            if "status" in fields:
                if fields["status"] == JobStatus.QUEUED.value:
                    self._queued_job_ids[job_id] = None
                else:
                    self._queued_job_ids.pop(job_id, None)

    def claim_next_job(self, worker_id: str) -> JobRecord | None:
        """Claim the next queued job for a worker."""
        with self._lock:
            if not self._queued_job_ids:
                return None
            job_id = next(iter(self._queued_job_ids))
            self._queued_job_ids.pop(job_id)
            job = self._jobs[job_id]
            now = utc_now_iso()
            job["status"] = JobStatus.RUNNING.value
            job["started_at"] = now
            job["updated_at"] = now
            job["stage"] = "claimed"
            job["worker_id"] = worker_id
            return cast(JobRecord, dict(job))

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        """Mark a job as succeeded and store its result."""
        self.update_job(
            job_id,
            status=JobStatus.SUCCEEDED.value,
            result=result,
            completed_at=utc_now_iso(),
            stage="completed",
        )

    def fail_job(self, job_id: str, error: JobError) -> None:
        """Mark a job as failed and store its error details."""
        self.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error=error,
            completed_at=utc_now_iso(),
            stage="failed",
        )

    def purge_expired(self, cutoff_iso: str) -> int:
        """Delete completed jobs older than the cutoff timestamp."""
        with self._lock:
            to_delete = [
                job_id
                for job_id, job in self._jobs.items()
                if (completed_at := job.get("completed_at")) is not None
                and completed_at < cutoff_iso
                and job["status"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
                in {
                    JobStatus.SUCCEEDED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELED.value,
                }
            ]
            for job_id in to_delete:
                self._jobs.pop(job_id, None)
                self._queued_job_ids.pop(job_id, None)
            return len(to_delete)

    def close(self) -> None:
        """Release resources held by the in-memory store."""
        return None


def _row_to_record(row: tuple[Any, ...]) -> JobRecord:
    """Convert a database row tuple into a JobRecord dict."""
    (
        job_id,
        status,
        request_json,
        result_json,
        error_json,
        created_at,
        updated_at,
        started_at,
        completed_at,
        stage,
        progress_json,
        cancel_requested,
        worker_id,
    ) = row
    return {
        "job_id": job_id,
        "status": status,
        "request": json_loads(request_json),
        "result": json_loads(result_json),
        "error": json_loads(error_json),
        "created_at": created_at,
        "updated_at": updated_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "stage": stage,
        "progress": json_loads(progress_json),
        "cancel_requested": bool(cancel_requested),
        "worker_id": worker_id,
    }


ALLOWED_UPDATE_COLUMNS = frozenset(
    {
        "status",
        "request",
        "request_json",
        "result",
        "result_json",
        "error",
        "error_json",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
        "stage",
        "progress",
        "progress_json",
        "cancel_requested",
        "worker_id",
    }
)


def _to_update_clause(fields: dict[str, Any]) -> tuple[str, list[Any]]:
    """Convert update fields into a SQL clause and parameter list."""
    columns: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        if key not in ALLOWED_UPDATE_COLUMNS:
            raise ValueError(f"Invalid column name: {key}")
        db_key = key
        encoded_value = value
        if key in {"request", "result", "error", "progress"}:
            db_key = f"{key}_json"
            encoded_value = json_dumps(value)
        if key == "cancel_requested":
            encoded_value = 1 if value else 0
        columns.append(f"{db_key} = ?")
        values.append(encoded_value)
    return ", ".join(columns), values
