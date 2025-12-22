import time

# ruff: noqa: PLR2004
from typing import Any

import pytest

from prompt_compiler.api.app import create_app
from prompt_compiler.dto.models import ModelSchema
from prompt_compiler.jobs import service as job_service_module


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("JOB_STORE", "memory")
    monkeypatch.setenv("WORKER_ENABLED", "false")
    monkeypatch.setenv("WORKER_POLL_INTERVAL_MS", "10")
    monkeypatch.setenv("WORKER_CONCURRENCY", "1")
    monkeypatch.setenv("PROMPT_COMPILER_ENV", "test")

    def fake_compile(job: dict[str, Any], registry):
        request = job.get("request") or {}
        source_model = request.get("source_model", "gpt-4o-mini")
        target_model = request.get("target_model", "gemini-2.5-flash")
        source_provider = registry.get_model(source_model).provider.provider
        target_provider = registry.get_model(target_model).provider.provider
        model_schema = ModelSchema()
        return {
            "candidate_prompt": "compiled prompt",
            "feedback": None,
            "diff_summary": None,
            "scores": {
                "primary_intent_score": 0.9,
                "tone_voice_score": 0.8,
                "domain_context_score": None,
                "constraint_scores": {},
                "final_score": 0.9,
            },
            "models": {
                "source_model": model_schema.dump(registry.get_model(source_model)),
                "target_model": model_schema.dump(registry.get_model(target_model)),
            },
            "run_metadata": {
                "max_retries": request.get("max_retries", 1),
                "score_threshold": request.get("score_threshold", 0.8),
                "scoring_algo": request.get("scoring_algo", "weighted"),
                "source_provider": source_provider,
                "target_provider": target_provider,
                "role_overrides": request.get("role_overrides", {}),
                "requested": {
                    "max_retries": request.get("max_retries"),
                    "score_threshold": request.get("score_threshold"),
                    "scoring_algo": request.get("scoring_algo"),
                },
            },
        }

    monkeypatch.setattr(job_service_module, "run_compile_job", fake_compile)

    monkeypatch.setenv("WORKER_ENABLED", "true")
    app = create_app()
    app.testing = True
    return app.test_client()


def test_healthz(api_client):
    response = api_client.get("/healthz")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_scoring_algorithms(api_client):
    response = api_client.get("/v1/scoring-algorithms")
    assert response.status_code == 200
    names = {item["name"] for item in response.json["algorithms"]}
    assert {"weighted", "geometric", "penalty", "dynamic"}.issubset(names)


def test_models_list(api_client):
    response = api_client.get("/v1/models")
    assert response.status_code == 200
    assert len(response.json["models"]) > 0


def test_enqueue_job(api_client):
    payload = {
        "raw_prompt": "Summarize this",
        "source_model": "gpt-4o-mini",
        "target_model": "gemini-2.5-flash",
    }
    response = api_client.post("/v1/compile-jobs", json=payload)
    assert response.status_code == 202
    body = response.json
    assert body["status"] == "queued"
    assert "job_id" in body
    assert "status_url" in body
    assert "result_url" in body


def test_worker_completes_job(api_client):
    payload = {
        "raw_prompt": "Summarize this",
        "source_model": "gpt-4o-mini",
        "target_model": "gemini-2.5-flash",
    }
    response = api_client.post("/v1/compile-jobs", json=payload)
    assert response.status_code == 202
    job_id = response.json["job_id"]

    status = None
    for _ in range(50):
        status_response = api_client.get(f"/v1/compile-jobs/{job_id}")
        status = status_response.json["status"]
        if status == "succeeded":
            break
        time.sleep(0.02)

    assert status == "succeeded"
    result_response = api_client.get(f"/v1/compile-jobs/{job_id}/result")
    assert result_response.status_code == 200
    result_body = result_response.json
    assert result_body["job_id"] == job_id
    assert result_body["status"] == "succeeded"
    assert "result" in result_body
    assert "candidate_prompt" in result_body["result"]


def test_worker_respects_request_options(api_client):
    payload = {
        "raw_prompt": "Summarize this with options",
        "source_model": "gpt-4o-mini",
        "target_model": "gemini-2.5-flash",
        "max_retries": 2,
        "score_threshold": 0.95,
        "scoring_algo": "penalty",
        "role_overrides": {"user": "You are a poet"},
    }
    response = api_client.post("/v1/compile-jobs", json=payload)
    assert response.status_code == 202
    job_id = response.json["job_id"]

    status = None
    for _ in range(50):
        status_response = api_client.get(f"/v1/compile-jobs/{job_id}")
        status = status_response.json["status"]
        if status == "succeeded":
            break
        time.sleep(0.02)

    assert status == "succeeded"
    result_response = api_client.get(f"/v1/compile-jobs/{job_id}/result")
    assert result_response.status_code == 200
    run_metadata = result_response.json["result"]["run_metadata"]
    assert run_metadata["max_retries"] == 2
    assert run_metadata["score_threshold"] == 0.95
    assert run_metadata["scoring_algo"] == "penalty"
    assert run_metadata["role_overrides"] == {"user": "You are a poet"}
    assert run_metadata["requested"] == {
        "max_retries": 2,
        "score_threshold": 0.95,
        "scoring_algo": "penalty",
    }


def test_worker_result_contains_expected_scores(api_client):
    payload = {
        "raw_prompt": "Summarize this with scores",
        "source_model": "gpt-4o-mini",
        "target_model": "gemini-2.5-flash",
    }
    response = api_client.post("/v1/compile-jobs", json=payload)
    assert response.status_code == 202
    job_id = response.json["job_id"]

    status = None
    for _ in range(50):
        status_response = api_client.get(f"/v1/compile-jobs/{job_id}")
        status = status_response.json["status"]
        if status == "succeeded":
            break
        time.sleep(0.02)

    assert status == "succeeded"
    result_response = api_client.get(f"/v1/compile-jobs/{job_id}/result")
    assert result_response.status_code == 200
    result = result_response.json["result"]
    scores = result["scores"]
    assert scores["primary_intent_score"] == 0.9
    assert scores["tone_voice_score"] == 0.8
    assert scores["final_score"] == 0.9
    assert "constraint_scores" in scores
    assert result["candidate_prompt"] == "compiled prompt"


def test_version_endpoint(api_client):
    response = api_client.get("/v1/version")
    assert response.status_code == 200
    assert "version" in response.json
    assert response.json["environment"] == "test"


def test_missing_job_returns_not_found(api_client):
    response = api_client.get("/v1/compile-jobs/missing-job")
    assert response.status_code == 404
    assert response.json["error"]["code"] == "not_found"


def test_cancel_missing_job_returns_not_found(api_client):
    response = api_client.delete("/v1/compile-jobs/missing-job")
    assert response.status_code == 404
    assert response.json["error"]["code"] == "not_found"


def test_register_model_and_filter_list(api_client):
    payload = {
        "provider": {"provider": "custom", "provider_type": "api", "metadata": {}},
        "model_name": "custom-model",
        "supports_system_messages": True,
        "context_window_size": 4096,
        "prompt_style": "markdown",
        "supports_json_mode": False,
        "prompting_tips": "Be concise.",
        "metadata": {},
    }
    response = api_client.post("/v1/models", json=payload)
    assert response.status_code == 200
    response = api_client.get("/v1/models?provider=custom&supports_json_mode=false")
    assert response.status_code == 200
    names = {model["model_name"] for model in response.json["models"]}
    assert "custom-model" in names
