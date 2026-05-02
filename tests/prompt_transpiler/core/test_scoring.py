import json
from unittest.mock import AsyncMock, patch

import pytest

from prompt_transpiler.core.exceptions import EvaluationError
from prompt_transpiler.core.scoring import (
    TIE_SCORE,
    LLMAdjudicator,
    PairwisePreferenceAlgorithm,
    WeightedScoreAlgorithm,
)
from prompt_transpiler.dto.models import (
    LLMResponse,
    Message,
    Model,
    ModelProviderType,
    PromptPayload,
    PromptStyle,
    Provider,
    TokenUsage,
)
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt, OriginalPrompt


@pytest.fixture
def mock_model():
    return Model(
        provider=Provider(provider="openai", provider_type=ModelProviderType.API),
        model_name="gpt-4",
        supports_system_messages=True,
        context_window_size=8192,
        prompt_style=PromptStyle.MARKDOWN,
        supports_json_mode=True,
        prompting_tips="tips",
    )


def test_weighted_score_algorithm(mock_model):
    algo = WeightedScoreAlgorithm(intent_weight=0.5, tone_weight=0.3, constraint_weight=0.2)
    original = OriginalPrompt(
        payload=PromptPayload(messages=[Message(role="user", content="orig")]), model=mock_model
    )
    candidate = CandidatePrompt(
        payload=PromptPayload(messages=[Message(role="user", content="cand")]), model=mock_model
    )

    # No scores yet
    assert algo.calculate_score(candidate, original) == 0.0

    # Set scores
    candidate.primary_intent_score = 1.0
    candidate.tone_voice_score = 0.8
    candidate.constraint_scores = {"c1": 1.0, "c2": 0.5}  # avg 0.75

    # Calculation: 1.0*0.5 + 0.8*0.3 + 0.75*0.2 = 0.5 + 0.24 + 0.15 = 0.89
    score = algo.calculate_score(candidate, original)
    assert score == pytest.approx(0.89)


def test_pairwise_preference_algorithm(mock_model):
    algo = PairwisePreferenceAlgorithm(intent_weight=0.5, tone_weight=0.3, constraint_weight=0.2)
    original = OriginalPrompt(
        payload=PromptPayload(messages=[Message(role="user", content="orig")]), model=mock_model
    )
    candidate = CandidatePrompt(
        payload=PromptPayload(messages=[Message(role="user", content="cand")]), model=mock_model
    )
    candidate.primary_intent_verdict = "candidate"
    candidate.tone_voice_verdict = "tie"
    candidate.constraint_verdicts = {"json": "baseline", "format": "candidate"}

    score = algo.calculate_score(candidate, original)
    assert score == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_llm_adjudicator_success(mock_model):
    with patch("prompt_transpiler.core.scoring.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        response_data = {
            "primary_intent_verdict": "candidate",
            "primary_intent_confidence": "strong",
            "tone_voice_verdict": "tie",
            "tone_voice_confidence": "weak",
            "constraint_verdicts": [
                {"constraint": "c1", "verdict": "candidate", "confidence": "medium"}
            ],
            "feedback_hint": "Good job",
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(response_data),
            model_name="gpt-4o",
            usage=TokenUsage(total_tokens=100),
        )
        mock_get_provider.return_value = mock_provider

        judge = LLMAdjudicator()
        original = OriginalPrompt(
            payload=PromptPayload(messages=[Message(role="user", content="orig")]),
            model=mock_model,
            response="base",
        )
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="cand")]),
            model=mock_model,
            response="cand_resp",
        )

        score = await judge.evaluate(candidate, original)

        assert score == 0.0
        assert candidate.primary_intent_verdict == "candidate"
        assert candidate.primary_intent_score == 1.0
        assert candidate.tone_voice_verdict == "tie"
        assert candidate.tone_voice_score == TIE_SCORE
        assert candidate.primary_intent_confidence == "strong"
        assert candidate.constraint_scores == {"c1": 1.0}
        assert candidate.constraint_verdicts == {"c1": "candidate"}
        assert candidate.constraint_confidences == {"c1": "medium"}
        assert candidate.feedback == "Good job"
        mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_llm_adjudicator_invalid_json(mock_model):
    with patch("prompt_transpiler.core.scoring.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = LLMResponse(
            content="Not JSON", model_name="gpt-4o", usage=TokenUsage(total_tokens=100)
        )
        mock_get_provider.return_value = mock_provider

        judge = LLMAdjudicator()
        original = OriginalPrompt(
            payload=PromptPayload(messages=[Message(role="user", content="orig")]), model=mock_model
        )
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="cand")]), model=mock_model
        )

        with pytest.raises(EvaluationError, match="Judge returned invalid JSON"):
            await judge.evaluate(candidate, original)


@pytest.mark.asyncio
async def test_llm_adjudicator_malformed_json_types(mock_model):
    """
    Test edge case where LLM returns valid JSON but the types are completely wrong
    (e.g. lists or booleans instead of strings). The Adjudicator should degrade gracefully
    and return default TIE_SCOREs instead of raising TypeError.
    """
    with patch("prompt_transpiler.core.scoring.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        # Return valid JSON but with invalid data types for expected strings
        response_data = {
            "primary_intent_verdict": ["not", "a", "string"],
            "primary_intent_confidence": True,
            "tone_voice_verdict": {"wrong": "type"},
            "tone_voice_confidence": 123,
            "constraint_verdicts": [{"constraint": ["bad"], "verdict": False, "confidence": None}],
            "feedback_hint": ["also", "wrong"],
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(response_data),
            model_name="gpt-4o",
            usage=TokenUsage(total_tokens=100),
        )
        mock_get_provider.return_value = mock_provider

        judge = LLMAdjudicator()
        original = OriginalPrompt(
            payload=PromptPayload(messages=[Message(role="user", content="orig")]),
            model=mock_model,
        )
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="cand")]),
            model=mock_model,
        )

        score = await judge.evaluate(candidate, original)

        assert score == 0.0
        assert candidate.primary_intent_verdict is None
        assert candidate.primary_intent_score == TIE_SCORE
        assert candidate.tone_voice_verdict is None
        assert candidate.tone_voice_score == TIE_SCORE
        assert candidate.primary_intent_confidence is None
        assert candidate.constraint_scores == {}
        assert candidate.constraint_verdicts == {}
        assert candidate.constraint_confidences is None


@pytest.mark.asyncio
async def test_llm_adjudicator_failure(mock_model):
    with patch("prompt_transpiler.core.scoring.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("LLM Error")
        mock_get_provider.return_value = mock_provider

        judge = LLMAdjudicator()
        original = OriginalPrompt(
            payload=PromptPayload(messages=[Message(role="user", content="orig")]), model=mock_model
        )
        candidate = CandidatePrompt(
            payload=PromptPayload(messages=[Message(role="user", content="cand")]), model=mock_model
        )

        score = await judge.evaluate(candidate, original)
        assert score == 0.0
        # Should log error but not raise, based on implementation
