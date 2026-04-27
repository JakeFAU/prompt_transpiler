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


def test_cancel_terminal_state_job_is_noop(mocker):
    """Verify that cancelling a job already in a terminal state (e.g. SUCCEEDED)
    does not mutate the state or throw an error. Mocks the underlying store."""
    mock_store = mocker.Mock()
    # Mock get_job to return a job that is already in a terminal state
    mock_store.get_job.return_value = {"job_id": "job-123", "status": JobStatus.SUCCEEDED.value}

    service = JobService(mock_store, ModelRegistry())
    job = service.cancel("job-123")

    # Assert that no update_job call was made because the state was terminal
    mock_store.update_job.assert_not_called()
    assert job is not None
    assert job["status"] == JobStatus.SUCCEEDED.value
