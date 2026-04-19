# ruff: noqa: PLR0913

from typing import Any, cast

from prompt_transpiler.core.pipeline import PromptTranspilerPipeline
from prompt_transpiler.core.registry import ModelRegistry
from prompt_transpiler.core.scoring import get_scoring_algorithm
from prompt_transpiler.dto.models import Message, PromptPayload
from prompt_transpiler.jobs.service import run_compile_job
from prompt_transpiler.llm.prompts.prompt_objects import (
    CandidatePrompt,
    CompilationAttempt,
    OriginalPrompt,
)


def test_run_compile_job_builds_response(monkeypatch):
    registry = ModelRegistry()
    source_model_name = "gpt-4o-mini"
    target_model_name = "gemini-2.5-flash"
    source_model = registry.get_model(source_model_name)
    target_model = registry.get_model(target_model_name)

    candidate = CandidatePrompt(
        payload=PromptPayload(messages=[Message(role="user", content="compiled")]),
        model=target_model,
    )
    candidate.primary_intent_score = 0.9
    candidate.tone_voice_score = 0.8
    candidate.constraint_scores = {"rule": 0.95}
    candidate.feedback = "ok"
    candidate.diff_summary = "diff"
    candidate.attempt_history = [
        CompilationAttempt(
            attempt=1,
            final_score=0.89,
            primary_intent_score=0.9,
            tone_voice_score=0.8,
            constraint_scores={"rule": 0.95},
            feedback="ok",
            accepted=True,
            new_best=True,
        )
    ]

    algo = get_scoring_algorithm("weighted")
    original = OriginalPrompt(
        payload=PromptPayload(messages=[Message(role="user", content="raw")]), model=source_model
    )
    candidate.total_score(algo, original)

    async def fake_run(
        self, raw_prompt, source_model, target_model, source_provider, target_provider, max_retries
    ):
        return candidate

    monkeypatch.setattr(PromptTranspilerPipeline, "run", fake_run)

    job = {
        "job_id": "job-1",
        "request": {
            "raw_prompt": "raw",
            "source_model": source_model_name,
            "target_model": target_model_name,
            "max_retries": 1,
            "score_threshold": 0.8,
            "scoring_algo": "weighted",
            "role_overrides": {"architect_provider": "openai"},
        },
    }

    result = run_compile_job(cast(Any, job), registry)
    assert result["transpiled_prompt"] == "user: compiled"
    assert result["scores"]["final_score"] is not None
    assert result["models"]["source_model"]["model_name"] == source_model_name
    assert result["models"]["target_model"]["model_name"] == target_model_name
    assert result["attempts"][0]["attempt"] == 1
