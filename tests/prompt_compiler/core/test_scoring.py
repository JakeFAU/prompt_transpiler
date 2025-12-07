import json
from unittest.mock import AsyncMock, patch

import pytest

from prompt_compiler.core.exceptions import EvaluationError
from prompt_compiler.core.scoring import LLMAdjudicator, WeightedScoreAlgorithm
from prompt_compiler.dto.models import Model, ModelProviderType, PromptStyle, Provider
from prompt_compiler.llm.prompts.prompt_objects import CandidatePrompt, OriginalPrompt

# Test constants
EXPECTED_INTENT_SCORE = 0.9
EXPECTED_TONE_SCORE = 0.8


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
    original = OriginalPrompt(prompt="orig", model=mock_model)
    candidate = CandidatePrompt(prompt="cand", model=mock_model)

    # No scores yet
    assert algo.calculate_score(candidate, original) == 0.0

    # Set scores
    candidate.primary_intent_score = 1.0
    candidate.tone_voice_score = 0.8
    candidate.constraint_scores = {"c1": 1.0, "c2": 0.5}  # avg 0.75

    # Calculation: 1.0*0.5 + 0.8*0.3 + 0.75*0.2 = 0.5 + 0.24 + 0.15 = 0.89
    score = algo.calculate_score(candidate, original)
    assert score == pytest.approx(0.89)


@pytest.mark.asyncio
async def test_llm_adjudicator_success(mock_model):
    with patch("prompt_compiler.core.scoring.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        response_data = {
            "primary_intent_score": 0.9,
            "tone_voice_score": 0.8,
            "constraint_scores": [{"constraint": "c1", "score": 1.0}],
            "feedback_hint": "Good job",
        }
        mock_provider.generate.return_value = json.dumps(response_data)
        mock_get_provider.return_value = mock_provider

        judge = LLMAdjudicator()
        original = OriginalPrompt(prompt="orig", model=mock_model, response="base")
        candidate = CandidatePrompt(prompt="cand", model=mock_model, response="cand_resp")

        score = await judge.evaluate(candidate, original)

        assert score == 0.0
        assert candidate.primary_intent_score == EXPECTED_INTENT_SCORE
        assert candidate.tone_voice_score == EXPECTED_TONE_SCORE
        assert candidate.constraint_scores == {"c1": 1.0}
        assert candidate.feedback == "Good job"
        mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_llm_adjudicator_invalid_json(mock_model):
    with patch("prompt_compiler.core.scoring.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.return_value = "Not JSON"
        mock_get_provider.return_value = mock_provider

        judge = LLMAdjudicator()
        original = OriginalPrompt(prompt="orig", model=mock_model)
        candidate = CandidatePrompt(prompt="cand", model=mock_model)

        with pytest.raises(EvaluationError, match="Judge returned invalid JSON"):
            await judge.evaluate(candidate, original)


@pytest.mark.asyncio
async def test_llm_adjudicator_failure(mock_model):
    with patch("prompt_compiler.core.scoring.get_llm_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate.side_effect = Exception("LLM Error")
        mock_get_provider.return_value = mock_provider

        judge = LLMAdjudicator()
        original = OriginalPrompt(prompt="orig", model=mock_model)
        candidate = CandidatePrompt(prompt="cand", model=mock_model)

        score = await judge.evaluate(candidate, original)
        assert score == 0.0
        # Should log error but not raise, based on implementation
