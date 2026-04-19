"""
Background worker for processing transpile jobs.

This module provides a threaded worker pool that continuously polls for queued
jobs and processes them asynchronously. Key features:

- Configurable concurrency (multiple worker threads)
- Graceful shutdown via WorkerController.stop()
- Automatic job retention/cleanup based on configured hours
- Poll interval throttling to reduce database load

Usage:
    controller = start_worker(job_service, poll_interval_ms=500, concurrency=2)
    # Later...
    controller.stop()
"""

import threading
from datetime import UTC, datetime, timedelta

from prompt_transpiler.jobs.service import JobService
from prompt_transpiler.jobs.util import sleep_ms
from prompt_transpiler.utils.logging import get_logger

logger = get_logger(__name__)


class WorkerController:
    """Lifecycle controller for background worker threads."""

    def __init__(
        self, service: JobService, poll_interval_ms: int, retention_hours: int | None
    ) -> None:
        self._service = service
        self._poll_interval_ms = poll_interval_ms
        self._retention_hours = retention_hours
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal worker threads to stop polling."""
        self._stop_event.set()

    def should_stop(self) -> bool:
        """Return True when a stop has been requested."""
        return self._stop_event.is_set()


def start_worker(
    service: JobService,
    poll_interval_ms: int,
    concurrency: int = 1,
    retention_hours: int | None = None,
) -> WorkerController:
    """Start background worker threads and return a controller."""
    controller = WorkerController(service, poll_interval_ms, retention_hours)
    for index in range(concurrency):
        worker_id = f"worker-{index + 1}"
        thread = threading.Thread(
            target=_worker_loop,
            args=(controller, worker_id),
            daemon=True,
        )
        thread.start()
    return controller


def _worker_loop(controller: WorkerController, worker_id: str) -> None:
    logger.info("Worker started", worker_id=worker_id)
    while not controller.should_stop():
        _maybe_purge(controller)
        job = controller._service.claim_next(worker_id)
        if job:
            controller._service.run_job(job)
            continue
        sleep_ms(controller._poll_interval_ms)
    logger.info("Worker stopped", worker_id=worker_id)


def _maybe_purge(controller: WorkerController) -> None:
    if controller._retention_hours is None:
        return
    cutoff = datetime.now(UTC) - timedelta(hours=controller._retention_hours)
    cutoff_iso = cutoff.isoformat()
    try:
        controller._service.store.purge_expired(cutoff_iso)
    except Exception as exc:
        logger.warning("Failed to purge expired jobs", error=str(exc))
