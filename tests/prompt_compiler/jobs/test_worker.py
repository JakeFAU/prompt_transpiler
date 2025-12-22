import time

from prompt_compiler.core.registry import ModelRegistry
from prompt_compiler.jobs.service import JobService
from prompt_compiler.jobs.store import MemoryJobStore
from prompt_compiler.jobs.worker import start_worker


def test_worker_processes_job():
    store = MemoryJobStore()
    registry = ModelRegistry()

    def fake_compile(job, registry):
        return {"candidate_prompt": "ok"}

    service = JobService(store, registry, compiler=fake_compile)
    job_id = service.enqueue_compile(
        {"raw_prompt": "hello", "source_model": "gpt-4o-mini", "target_model": "gemini-2.5-flash"}
    )

    controller = start_worker(service, poll_interval_ms=5, concurrency=1)
    try:
        for _ in range(100):
            job = store.get_job(job_id)
            if job and job.get("status") == "succeeded":
                break
            time.sleep(0.01)
        job = store.get_job(job_id)
        assert job is not None
        assert job.get("status") == "succeeded"
    finally:
        controller.stop()
