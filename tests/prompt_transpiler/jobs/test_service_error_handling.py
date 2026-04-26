from prompt_transpiler.core.registry import ModelRegistry
from prompt_transpiler.jobs.models import JobStatus
from prompt_transpiler.jobs.service import JobService
from prompt_transpiler.jobs.store import MemoryJobStore


def test_run_job_compiler_exception_failure():
    """
    Test that if the compiler raises an unexpected exception (e.g. from network timeout),
    the JobService catches it and properly fails the job instead of crashing the worker.
    """
    store = MemoryJobStore()
    registry = ModelRegistry()

    def fake_compiler(job, registry):
        raise RuntimeError("External API connection timed out.")

    service = JobService(store, registry, compiler=fake_compiler)

    job_id = service.enqueue_compile({"raw_prompt": "hello"})
    job = store.claim_next_job("worker-1")
    assert job is not None

    service.run_job(job)

    failed_job = store.get_job(job_id)
    assert failed_job is not None
    assert failed_job["status"] == JobStatus.FAILED.value
    assert failed_job["stage"] == "failed"

    error = failed_job.get("error")
    assert error is not None
    assert error["code"] == "compile_failed"
    assert error["message"] == "Transpilation failed"
    assert "External API connection timed out." in error["details"]["error"]
