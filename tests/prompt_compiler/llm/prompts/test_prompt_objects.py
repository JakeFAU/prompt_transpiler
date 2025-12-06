import pytest

from prompt_complier.dto.models import Model, ModelProviderType, PromptStyle, Provider
from prompt_complier.llm.prompts.prompt_objects import (
    CandidatePrompt,
    OriginalPrompt,
    OriginalPromptSchema,
    ScoringAlgorithm,
)

# Test constants
EXPECTED_SCORE = 0.95


@pytest.fixture
def mock_model():
    provider = Provider(provider="OpenAI", provider_type=ModelProviderType.API, metadata={})
    return Model(
        provider=provider,
        model_name="gpt-4",
        supports_system_messages=True,
        context_window_size=8192,
        prompt_style=PromptStyle.MARKDOWN,
        supports_json_mode=True,
        prompting_tips="Be concise.",
        metadata={},
    )


def test_original_prompt_creation(mock_model):
    op = OriginalPrompt(
        prompt="Hello", model=mock_model, response_format={"type": "json"}, response="Hi"
    )
    assert op.prompt == "Hello"
    assert op.model == mock_model
    assert op.response_format == {"type": "json"}
    assert op.response == "Hi"


def test_original_prompt_schema(mock_model):
    schema = OriginalPromptSchema()
    data = {
        "prompt": "Hello",
        "model": {
            "provider": {"provider": "OpenAI", "provider_type": "api", "metadata": {}},
            "model_name": "gpt-4",
            "supports_system_messages": True,
            "context_window_size": 8192,
            "prompt_style": "markdown",
            "supports_json_mode": True,
            "prompting_tips": "Be concise.",
            "metadata": {},
        },
        "response_format": {"type": "json"},
        "response": "Hi",
    }
    op = schema.load(data)
    assert isinstance(op, OriginalPrompt)
    assert op.prompt == "Hello"


def test_candidate_prompt_creation(mock_model):
    cp = CandidatePrompt(
        prompt="Hello optimized",
        model=mock_model,
        response_format={"type": "json"},
        response="Hi optimized",
    )
    assert cp.prompt == "Hello optimized"
    assert cp.model == mock_model
    assert cp.primary_intent_score is None
    assert cp._cached_score is None


class MockScoringAlgorithm(ScoringAlgorithm):
    def calculate_score(self, candidate: "CandidatePrompt", original: OriginalPrompt) -> float:
        return EXPECTED_SCORE


def test_candidate_prompt_scoring_caching(mock_model):
    op = OriginalPrompt(prompt="Hello", model=mock_model)
    cp = CandidatePrompt(prompt="Hello optimized", model=mock_model)
    algo = MockScoringAlgorithm()

    # First calculation
    score1 = cp.total_score(algo, op)
    assert score1 == EXPECTED_SCORE
    assert cp._cached_score == EXPECTED_SCORE
    assert cp._cached_algo_id == id(algo)

    # Second calculation (should hit cache)
    # We can verify this by mocking the calculate_score method if we used a mock object,
    # but here we rely on the internal state check.
    score2 = cp.total_score(algo, op)
    assert score2 == EXPECTED_SCORE

    # Different algo instance
    algo2 = MockScoringAlgorithm()
    score3 = cp.total_score(algo2, op)
    assert score3 == EXPECTED_SCORE
    assert cp._cached_algo_id == id(algo2)
