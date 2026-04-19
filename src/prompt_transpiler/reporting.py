"""Helpers for rendering transpile results as structured reports."""

from typing import Any

from prompt_transpiler.dto.models import ModelSchema
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt, CompilationAttempt


def build_transpile_report(
    candidate: CandidatePrompt,
    *,
    source_model: Any,
    target_model: Any,
    final_score: float,
    token_usage: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Build a stable machine-readable transpile report."""
    model_schema = ModelSchema()
    attempts = getattr(candidate, "attempt_history", [])
    if not isinstance(attempts, list):
        attempts = []
    run_metadata = getattr(candidate, "run_metadata", {})
    if not isinstance(run_metadata, dict):
        run_metadata = {}
    return {
        "transpiled_prompt": getattr(candidate, "prompt", ""),
        "candidate_prompt": getattr(candidate, "prompt", ""),
        "feedback": getattr(candidate, "feedback", None),
        "diff_summary": getattr(candidate, "diff_summary", None),
        "scores": {
            "primary_intent_score": getattr(candidate, "primary_intent_score", None),
            "tone_voice_score": getattr(candidate, "tone_voice_score", None),
            "domain_context_score": getattr(candidate, "domain_context_score", None),
            "constraint_scores": getattr(candidate, "constraint_scores", None),
            "final_score": final_score,
        },
        "comparisons": {
            "primary_intent_verdict": getattr(candidate, "primary_intent_verdict", None),
            "primary_intent_confidence": getattr(candidate, "primary_intent_confidence", None),
            "tone_voice_verdict": getattr(candidate, "tone_voice_verdict", None),
            "tone_voice_confidence": getattr(candidate, "tone_voice_confidence", None),
            "constraint_verdicts": getattr(candidate, "constraint_verdicts", None),
            "constraint_confidences": getattr(candidate, "constraint_confidences", None),
        },
        "models": {
            "source_model": model_schema.dump(source_model),
            "target_model": model_schema.dump(target_model),
        },
        "run_metadata": run_metadata,
        "attempts": [_dump_attempt(attempt) for attempt in attempts],
        "token_usage": token_usage or {},
    }


build_compile_report = build_transpile_report


def _dump_attempt(attempt: CompilationAttempt) -> dict[str, Any]:
    return {
        "attempt": attempt.attempt,
        "final_score": attempt.final_score,
        "primary_intent_score": attempt.primary_intent_score,
        "tone_voice_score": attempt.tone_voice_score,
        "constraint_scores": attempt.constraint_scores,
        "primary_intent_verdict": attempt.primary_intent_verdict,
        "tone_voice_verdict": attempt.tone_voice_verdict,
        "constraint_verdicts": attempt.constraint_verdicts,
        "primary_intent_confidence": attempt.primary_intent_confidence,
        "tone_voice_confidence": attempt.tone_voice_confidence,
        "constraint_confidences": attempt.constraint_confidences,
        "feedback": attempt.feedback,
        "accepted": attempt.accepted,
        "new_best": attempt.new_best,
    }
