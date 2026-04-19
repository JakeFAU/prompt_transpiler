import pytest

from prompt_transpiler.reporting import build_transpile_report, build_compile_report
from prompt_transpiler.dto.models import Model, PromptPayload, Message
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt, CompilationAttempt

@pytest.fixture
def test_payload():
    return PromptPayload(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!")
        ]
    )

@pytest.fixture
def minimal_candidate(model_obj, test_payload):
    return CandidatePrompt(
        payload=test_payload,
        model=model_obj
    )

@pytest.fixture
def full_candidate(model_obj, test_payload):
    attempt1 = CompilationAttempt(
        attempt=1,
        final_score=0.8,
        primary_intent_score=0.9,
        tone_voice_score=0.8,
        constraint_scores={"len": 0.7},
        primary_intent_verdict="Good",
        tone_voice_verdict="Okay",
        constraint_verdicts={"len": "Passable"},
        primary_intent_confidence="High",
        tone_voice_confidence="Medium",
        constraint_confidences={"len": "Low"},
        feedback="Make it better",
        accepted=True,
        new_best=True
    )

    candidate = CandidatePrompt(
        payload=test_payload,
        model=model_obj,
        response="I can help you with that.",
        primary_intent_score=0.95,
        tone_voice_score=0.85,
        domain_context_score=0.9,
        constraint_scores={"c1": 0.8},
        primary_intent_verdict="Excellent",
        tone_voice_verdict="Good",
        constraint_verdicts={"c1": "Passed"},
        primary_intent_confidence="Very High",
        tone_voice_confidence="High",
        constraint_confidences={"c1": "Medium"},
        feedback="Great job",
        diff_summary="Changed a few words",
        attempt_history=[attempt1],
        run_metadata={"duration": 1.2}
    )
    return candidate

def test_build_transpile_report_basic(minimal_candidate, model_obj):
    report = build_transpile_report(
        candidate=minimal_candidate,
        source_model=model_obj,
        target_model=model_obj,
        final_score=0.5
    )

    assert report["transpiled_prompt"] == minimal_candidate.prompt
    assert report["candidate_prompt"] == minimal_candidate.prompt
    assert report["feedback"] is None
    assert report["diff_summary"] is None

    scores = report["scores"]
    assert scores["primary_intent_score"] is None
    assert scores["tone_voice_score"] is None
    assert scores["domain_context_score"] is None
    assert scores["constraint_scores"] is None
    assert scores["final_score"] == 0.5

    comparisons = report["comparisons"]
    assert comparisons["primary_intent_verdict"] is None
    assert comparisons["primary_intent_confidence"] is None
    assert comparisons["tone_voice_verdict"] is None
    assert comparisons["tone_voice_confidence"] is None
    assert comparisons["constraint_verdicts"] is None
    assert comparisons["constraint_confidences"] is None

    models = report["models"]
    assert "source_model" in models
    assert models["source_model"]["model_name"] == model_obj.model_name
    assert "target_model" in models
    assert models["target_model"]["model_name"] == model_obj.model_name

    assert report["run_metadata"] == {}
    assert report["attempts"] == []
    assert report["token_usage"] == {}

def test_build_transpile_report_full(full_candidate, model_obj):
    report = build_transpile_report(
        candidate=full_candidate,
        source_model=model_obj,
        target_model=model_obj,
        final_score=0.9,
        token_usage={"total": {"prompt_tokens": 10}}
    )

    assert report["transpiled_prompt"] == full_candidate.prompt
    assert report["candidate_prompt"] == full_candidate.prompt
    assert report["feedback"] == "Great job"
    assert report["diff_summary"] == "Changed a few words"

    scores = report["scores"]
    assert scores["primary_intent_score"] == 0.95
    assert scores["tone_voice_score"] == 0.85
    assert scores["domain_context_score"] == 0.9
    assert scores["constraint_scores"] == {"c1": 0.8}
    assert scores["final_score"] == 0.9

    comparisons = report["comparisons"]
    assert comparisons["primary_intent_verdict"] == "Excellent"
    assert comparisons["primary_intent_confidence"] == "Very High"
    assert comparisons["tone_voice_verdict"] == "Good"
    assert comparisons["tone_voice_confidence"] == "High"
    assert comparisons["constraint_verdicts"] == {"c1": "Passed"}
    assert comparisons["constraint_confidences"] == {"c1": "Medium"}

    assert report["run_metadata"] == {"duration": 1.2}

    assert len(report["attempts"]) == 1
    attempt = report["attempts"][0]
    assert attempt["attempt"] == 1
    assert attempt["final_score"] == 0.8
    assert attempt["primary_intent_score"] == 0.9
    assert attempt["tone_voice_score"] == 0.8
    assert attempt["constraint_scores"] == {"len": 0.7}
    assert attempt["primary_intent_verdict"] == "Good"
    assert attempt["tone_voice_verdict"] == "Okay"
    assert attempt["constraint_verdicts"] == {"len": "Passable"}
    assert attempt["primary_intent_confidence"] == "High"
    assert attempt["tone_voice_confidence"] == "Medium"
    assert attempt["constraint_confidences"] == {"len": "Low"}
    assert attempt["feedback"] == "Make it better"
    assert attempt["accepted"] is True
    assert attempt["new_best"] is True

    assert report["token_usage"] == {"total": {"prompt_tokens": 10}}


def test_build_transpile_report_defaults(minimal_candidate, model_obj):
    # Mess up the expected types directly by bypassing validators just for this test
    # (Since CandidatePrompt uses attrs with validators, we override the dict dict __dict__ or using patch)
    object.__setattr__(minimal_candidate, "attempt_history", None)
    object.__setattr__(minimal_candidate, "run_metadata", "not a dict")

    report = build_transpile_report(
        candidate=minimal_candidate,
        source_model=model_obj,
        target_model=model_obj,
        final_score=0.1
    )

    assert report["attempts"] == []
    assert report["run_metadata"] == {}


def test_build_compile_report_alias():
    assert build_compile_report is build_transpile_report
