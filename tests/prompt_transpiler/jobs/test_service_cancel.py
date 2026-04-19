from prompt_transpiler.core.registry import ModelRegistry
from prompt_transpiler.jobs.models import JobStatus
from prompt_transpiler.jobs.service import JobService
from prompt_transpiler.jobs.store import MemoryJobStore


def test_cancel_queued_job_marks_canceled():
    store = MemoryJobStore()
    service = JobService(store, ModelRegistry())
    job_id = service.enqueue_compile({"raw_prompt": "hello"})

    job = service.cancel(job_id)
    assert job is not None
    assert job["status"] == JobStatus.CANCELED.value


def test_cancel_running_job_sets_flag():
    store = MemoryJobStore()
    service = JobService(store, ModelRegistry())
    job_id = service.enqueue_compile({"raw_prompt": "hello"})
    store.update_job(job_id, status=JobStatus.RUNNING.value)

    job = service.cancel(job_id)
    assert job is not None
    assert job["cancel_requested"] is True
