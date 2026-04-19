import pytest

from prompt_transpiler.dto.models import (
    Message,
    Model,
    ModelProviderType,
    PromptPayload,
    PromptStyle,
    Provider,
)
from prompt_transpiler.llm.prompts.prompt_objects import (
    CandidatePrompt,
    CandidatePromptSchema,
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


@pytest.fixture
def mock_payload():
    return PromptPayload(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
        ],
        response_format={"type": "json"},
    )


def test_original_prompt_creation(mock_model, mock_payload):
    op = OriginalPrompt(payload=mock_payload, model=mock_model, response="Hi")
    assert op.prompt == mock_payload.full_text
    assert op.payload == mock_payload
    assert op.model == mock_model
    assert op.response == "Hi"
    assert op.response_format == mock_payload.response_format


def test_original_prompt_schema(mock_model, mock_payload):
    schema = OriginalPromptSchema()
    data = {
        "payload": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "response_format": {"type": "json"},
        },
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
        "response": "Hi",
    }
    op = schema.load(data)
    assert isinstance(op, OriginalPrompt)
    assert op.prompt == mock_payload.full_text


def test_transpiled_prompt_creation(mock_model, mock_payload):
    cp = CandidatePrompt(payload=mock_payload, model=mock_model, response="Hi optimized")
    assert cp.prompt == mock_payload.full_text
    assert cp.payload == mock_payload
    assert cp.model == mock_model
    assert cp.response_format == mock_payload.response_format
    assert cp.primary_intent_score is None
    assert cp.diff_summary is None
    assert cp._cached_score is None


class MockScoringAlgorithm(ScoringAlgorithm):
    def calculate_score(self, candidate: "CandidatePrompt", original: OriginalPrompt) -> float:
        return EXPECTED_SCORE


def test_transpiled_prompt_scoring_caching(mock_model, mock_payload):
    op = OriginalPrompt(payload=mock_payload, model=mock_model)
    cp = CandidatePrompt(payload=mock_payload, model=mock_model)
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


def test_candidate_prompt_schema(mock_model, mock_payload):
    schema = CandidatePromptSchema()
    primary_intent_score = 0.9
    final_score = 0.85
    data = {
        "payload": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "response_format": {"type": "json"},
        },
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
        "response": "Hi optimized",
        "primary_intent_score": primary_intent_score,
        "feedback": "Good job",
        "diff_summary": "No changes",
        "attempt_history": [
            {
                "attempt": 1,
                "final_score": final_score,
                "accepted": False,
                "new_best": False,
            }
        ],
        "run_metadata": {"foo": "bar"},
    }
    cp = schema.load(data)
    assert isinstance(cp, CandidatePrompt)
    assert cp.prompt == mock_payload.full_text
    assert cp.primary_intent_score == primary_intent_score
    assert cp.feedback == "Good job"
    assert cp.diff_summary == "No changes"
    assert len(cp.attempt_history) == 1
    assert cp.attempt_history[0].final_score == final_score
    assert cp.run_metadata == {"foo": "bar"}
