import pytest
from typing import Any, cast
from prompt_transpiler.jobs.service import JobService
from prompt_transpiler.jobs.models import JobRecord

def test_run_job_store_update_exception_bubbles_up(mocker: Any) -> None:
    """
    Edge Case: The job store raises an exception when attempting to update the job stage.
    Since this update happens outside the try...except block designed for the compiler,
    the exception should bubble up to prevent an inconsistent state machine.
    """
    mock_store = mocker.MagicMock()
    # Mock update_job to raise an external boundary error (e.g. database disconnect)
    mock_store.update_job.side_effect = ConnectionError("Database disconnected")

    mock_registry = mocker.MagicMock()
    mock_compiler = mocker.MagicMock()

    service = JobService(store=mock_store, registry=mock_registry, compiler=mock_compiler)
    job = cast(JobRecord, {"job_id": "job-123", "status": "running"})

    with pytest.raises(ConnectionError, match="Database disconnected"):
        service.run_job(job)

    # Ensure the compiler was never executed because the state machine update failed
    mock_compiler.assert_not_called()
    # Ensure fail_job was not called either, since it bubbles up
    mock_store.fail_job.assert_not_called()
