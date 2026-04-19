import json
import time

# ruff: noqa: PLR2004
from typing import Any, cast

import pytest
from flask.testing import FlaskClient
from pytest import MonkeyPatch

from prompt_transpiler.api.app import create_app
from prompt_transpiler.dto.models import ModelSchema
from prompt_transpiler.jobs import service as job_service_module


def _get_json(response: Any) -> dict[str, Any]:
    payload = response.get_json(silent=True)
    if payload is None:
        try:
            payload = json.loads(response.get_data(as_text=True))
        except json.JSONDecodeError as exc:
            pytest.fail(f"Expected JSON response, got: {response.get_data(as_text=True)!r}")
            raise exc
    return cast(dict[str, Any], payload)


@pytest.fixture
def api_client(monkeypatch: MonkeyPatch) -> FlaskClient:
    monkeypatch.setenv("JOB_STORE", "memory")
    monkeypatch.setenv("WORKER_ENABLED", "false")
    monkeypatch.setenv("WORKER_POLL_INTERVAL_MS", "10")
    monkeypatch.setenv("WORKER_CONCURRENCY", "1")
    monkeypatch.setenv("PROMPT_TRANSPILER_ENV", "test")

    def fake_compile(job: dict[str, Any], registry: Any) -> dict[str, Any]:
        request = job.get("request") or {}
        source_model = request.get("source_model", "gpt-4o-mini")
        target_model = request.get("target_model", "gemini-2.5-flash")
        source_provider = registry.get_model(source_model).provider.provider
        target_provider = registry.get_model(target_model).provider.provider
        model_schema = ModelSchema()
        return {
            "transpiled_prompt": "transpiled prompt",
            "feedback": None,
            "diff_summary": None,
            "scores": {
                "primary_intent_score": 0.9,
                "tone_voice_score": 0.8,
                "domain_context_score": None,
                "constraint_scores": {},
                "final_score": 0.9,
            },
            "comparisons": {
                "primary_intent_verdict": "candidate",
                "primary_intent_confidence": "medium",
                "tone_voice_verdict": "tie",
                "tone_voice_confidence": "weak",
                "constraint_verdicts": {},
                "constraint_confidences": {},
            },
            "models": {
                "source_model": model_schema.dump(registry.get_model(source_model)),
                "target_model": model_schema.dump(registry.get_model(target_model)),
            },
            "run_metadata": {
                "max_retries": request.get("max_retries", 1),
                "score_threshold": request.get("score_threshold", 0.8),
                "scoring_algo": request.get("scoring_algo", "pairwise"),
                "source_provider": source_provider,
                "target_provider": target_provider,
                "role_overrides": request.get("role_overrides", {}),
                "requested": {
                    "max_retries": request.get("max_retries"),
                    "score_threshold": request.get("score_threshold"),
                    "scoring_algo": request.get("scoring_algo"),
                },
            },
            "attempts": [
                {
                    "attempt": 1,
                    "final_score": 0.9,
                    "primary_intent_score": 0.9,
                    "tone_voice_score": 0.8,
                    "constraint_scores": {},
                    "primary_intent_verdict": "candidate",
                    "tone_voice_verdict": "tie",
                    "constraint_verdicts": {},
                    "primary_intent_confidence": "medium",
                    "tone_voice_confidence": "weak",
                    "constraint_confidences": {},
                    "feedback": None,
                    "accepted": True,
                    "new_best": True,
                }
            ],
            "token_usage": {},
        }

    monkeypatch.setattr(job_service_module, "run_compile_job", fake_compile)

    monkeypatch.setenv("WORKER_ENABLED", "true")
    app = create_app(start_worker_flag=True)
    app.testing = True
    return app.test_client()


def test_healthz(api_client: FlaskClient) -> None:
    response = api_client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_scoring_algorithms(api_client: FlaskClient) -> None:
    response = api_client.get("/v1/scoring-algorithms")
    assert response.status_code == 200
    payload = _get_json(response)
    names = {item["name"] for item in payload["algorithms"]}
    assert {"pairwise", "weighted", "geometric", "penalty", "dynamic"}.issubset(names)


def test_models_list(api_client: FlaskClient) -> None:
    response = api_client.get("/v1/models")
    assert response.status_code == 200
    payload = _get_json(response)
    assert len(payload["models"]) > 0


def test_enqueue_job(api_client: FlaskClient) -> None:
    payload = {
        "raw_prompt": "Summarize this",
        "source_model": "gpt-4o-mini",
        "target_model": "gemini-2.5-flash",
    }
    response = api_client.post("/v1/transpile-jobs", json=payload)
    assert response.status_code == 202
    body = _get_json(response)
    assert body["status"] == "queued"
    assert "job_id" in body
    assert "status_url" in body
    assert "result_url" in body


def test_worker_completes_job(api_client: FlaskClient) -> None:
    payload = {
        "raw_prompt": "Summarize this",
        "source_model": "gpt-4o-mini",
        "target_model": "gemini-2.5-flash",
    }
    response = api_client.post("/v1/transpile-jobs", json=payload)
    assert response.status_code == 202
    body = _get_json(response)
    job_id = body["job_id"]

    status = None
    for _ in range(50):
        status_response = api_client.get(f"/v1/transpile-jobs/{job_id}")
        status_payload = _get_json(status_response)
        status = status_payload["status"]
        if status == "succeeded":
            break
        time.sleep(0.02)

    assert status == "succeeded"
    result_response = api_client.get(f"/v1/transpile-jobs/{job_id}/result")
    assert result_response.status_code == 200
    result_body = _get_json(result_response)
    assert result_body["job_id"] == job_id
    assert result_body["status"] == "succeeded"
    assert "result" in result_body
    assert "transpiled_prompt" in result_body["result"]


def test_worker_respects_request_options(api_client: FlaskClient) -> None:
    payload = {
        "raw_prompt": "Summarize this with options",
        "source_model": "gpt-4o-mini",
        "target_model": "gemini-2.5-flash",
        "max_retries": 2,
        "score_threshold": 0.95,
        "scoring_algo": "penalty",
        "role_overrides": {"user": "You are a poet"},
    }
    response = api_client.post("/v1/transpile-jobs", json=payload)
    assert response.status_code == 202
    body = _get_json(response)
    job_id = body["job_id"]

    status = None
    for _ in range(50):
        status_response = api_client.get(f"/v1/transpile-jobs/{job_id}")
        status_payload = _get_json(status_response)
        status = status_payload["status"]
        if status == "succeeded":
            break
        time.sleep(0.02)

    assert status == "succeeded"
    result_response = api_client.get(f"/v1/transpile-jobs/{job_id}/result")
    assert result_response.status_code == 200
    result_payload = _get_json(result_response)
    run_metadata = result_payload["result"]["run_metadata"]
    assert run_metadata["max_retries"] == 2
    assert run_metadata["score_threshold"] == 0.95
    assert run_metadata["scoring_algo"] == "penalty"
    assert run_metadata["role_overrides"] == {"user": "You are a poet"}
    assert run_metadata["requested"] == {
        "max_retries": 2,
        "score_threshold": 0.95,
        "scoring_algo": "penalty",
    }


def test_worker_result_contains_expected_scores(api_client: FlaskClient) -> None:
    payload = {
        "raw_prompt": "Summarize this with scores",
        "source_model": "gpt-4o-mini",
        "target_model": "gemini-2.5-flash",
    }
    response = api_client.post("/v1/transpile-jobs", json=payload)
    assert response.status_code == 202
    body = _get_json(response)
    job_id = body["job_id"]

    status = None
    for _ in range(50):
        status_response = api_client.get(f"/v1/transpile-jobs/{job_id}")
        status_payload = _get_json(status_response)
        status = status_payload["status"]
        if status == "succeeded":
            break
        time.sleep(0.02)

    assert status == "succeeded"
    result_response = api_client.get(f"/v1/transpile-jobs/{job_id}/result")
    assert result_response.status_code == 200
    result_payload = _get_json(result_response)
    result = result_payload["result"]
    scores = result["scores"]
    assert scores["primary_intent_score"] == 0.9
    assert scores["tone_voice_score"] == 0.8
    assert scores["final_score"] == 0.9
    assert "constraint_scores" in scores
    assert result["comparisons"]["primary_intent_verdict"] == "candidate"
    assert result["attempts"][0]["accepted"] is True
    assert result["transpiled_prompt"] == "transpiled prompt"


def test_version_endpoint(api_client: FlaskClient) -> None:
    response = api_client.get("/v1/version")
    assert response.status_code == 200
    payload = _get_json(response)
    assert "version" in payload
    assert payload["environment"] == "test"


def test_validation_error_returns_400(api_client: FlaskClient) -> None:
    payload = {"source_model": "gpt-4o-mini", "target_model": "gemini-2.5-flash"}
    response = api_client.post("/v1/transpile-jobs", json=payload)
    assert response.status_code == 422
    body = _get_json(response)
    assert body["error"]["code"] == "http_error"


def test_missing_job_returns_not_found(api_client: FlaskClient) -> None:
    response = api_client.get("/v1/transpile-jobs/missing-job")
    assert response.status_code == 404
    body = _get_json(response)
    assert body["error"]["code"] == "not_found"


def test_cancel_missing_job_returns_not_found(api_client: FlaskClient) -> None:
    response = api_client.delete("/v1/transpile-jobs/missing-job")
    assert response.status_code == 404
    body = _get_json(response)
    assert body["error"]["code"] == "not_found"


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
    body = _get_json(response)
    names = {model["model_name"] for model in body["models"]}
    assert "custom-model" in names
