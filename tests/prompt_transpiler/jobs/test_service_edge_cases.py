import pytest

from prompt_transpiler.core.registry import ModelRegistry
from prompt_transpiler.jobs.service import run_compile_job


def test_run_compile_job_none_source_model_raises_attribute_error(mocker):
    """
    Edge Case: Transpile job receives a request payload where 'source_model' is explicitly None.
    Learning: The registry fallback logic in get_model attempts to call .lower() on the
    model_name. If the request unexpectedly provides None for the model, this raises an
    unhandled AttributeError, causing the job to fail with a stack trace instead of a
    clean validation error.
    """
    registry = ModelRegistry()

    # Mock external boundaries to isolate the specific logic being tested
    mocker.patch("prompt_transpiler.jobs.service.PromptTranspilerPipeline.run")

    job = {
        "job_id": "job-edge-1",
        "request": {
            "raw_prompt": "Make this more professional.",
            "source_model": None,  # The unexpectedly empty state
            "target_model": "gemini-2.5-flash",
        },
    }

    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'lower'"):
        run_compile_job(job, registry)  # type: ignore
