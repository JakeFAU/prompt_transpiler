from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prompt_compiler.core.interfaces import (
    IArchitect,
    IDecompiler,
    IHistorian,
    IJudge,
    IPilot,
)
from prompt_compiler.core.pipeline import (
    PromptCompilerPipeline,
    compile_pipeline,
)
from prompt_compiler.dto.models import (
    IntermediateRepresentation,
    IntermediateRepresentationData,
    IntermediateRepresentationMeta,
    IntermediateRepresentationSpec,
    Model,
    ModelProviderType,
    PromptStyle,
    Provider,
)
from prompt_compiler.llm.prompts.prompt_objects import (
    CandidatePrompt,
    OriginalPrompt,
    ScoringAlgorithm,
)

# Test constants
EXPECTED_RETRY_COUNT = 2
EXPECTED_EARLY_STOP_COUNT = 3


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


@pytest.fixture
def mock_ir(mock_model):
    return IntermediateRepresentation(
        meta=IntermediateRepresentationMeta(source_model=mock_model, target_model=mock_model),
        spec=IntermediateRepresentationSpec(
            primary_intent="intent",
            tone_voice="tone",
            domain_context="domain",
            constraints=[],
            input_format="text",
            output_schema="text",
        ),
        data=IntermediateRepresentationData(few_shot_examples=[]),
    )


@pytest.fixture
def mock_candidate_factory(mock_model):
    def _create():
        candidate = CandidatePrompt(prompt="candidate", model=mock_model)
        candidate.feedback = "feedback"
        return candidate

    return _create


@pytest.fixture
def mock_roles(mock_model, mock_ir, mock_candidate_factory):
    historian = MagicMock(spec=IHistorian)
    historian.establish_baseline = AsyncMock(
        return_value=OriginalPrompt(prompt="original", model=mock_model)
    )

    decompiler = MagicMock(spec=IDecompiler)
    decompiler.decompile = AsyncMock(return_value=mock_ir)

    architect = MagicMock(spec=IArchitect)
    architect.design_prompt = AsyncMock(
        side_effect=lambda *args, **kwargs: mock_candidate_factory()
    )

    pilot = MagicMock(spec=IPilot)
    pilot.test_candidate = AsyncMock(side_effect=lambda c: c)

    judge = MagicMock(spec=IJudge)
    judge.evaluate = AsyncMock(return_value=0.0)

    scoring = MagicMock(spec=ScoringAlgorithm)
    scoring.calculate_score = MagicMock(return_value=0.95)

    return historian, decompiler, architect, pilot, judge, scoring


@pytest.mark.asyncio
async def test_pipeline_success(mock_roles, mock_model):
    historian, decompiler, architect, pilot, judge, scoring = mock_roles

    pipeline = PromptCompilerPipeline(
        historian=historian,
        decompiler=decompiler,
        architect=architect,
        pilot=pilot,
        judge=judge,
        scoring_algorithm=scoring,
        score_threshold=0.9,
        max_retries=1,
    )

    result = await pipeline.run(raw_prompt="raw", source_model="gpt-4", target_model="gemini-pro")

    assert result.prompt == "candidate"
    historian.establish_baseline.assert_called_once()
    decompiler.decompile.assert_called_once()
    architect.design_prompt.assert_called_once()
    pilot.test_candidate.assert_called_once()
    judge.evaluate.assert_called_once()
    scoring.calculate_score.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_retry_loop(mock_roles, mock_model):
    historian, decompiler, architect, pilot, judge, scoring = mock_roles

    # First attempt fails threshold, second succeeds
    scoring.calculate_score.side_effect = [0.5, 0.95]

    pipeline = PromptCompilerPipeline(
        historian=historian,
        decompiler=decompiler,
        architect=architect,
        pilot=pilot,
        judge=judge,
        scoring_algorithm=scoring,
        score_threshold=0.9,
        max_retries=2,
        early_stop_patience=5,
    )

    result = await pipeline.run(raw_prompt="raw", source_model="gpt-4", target_model="gemini-pro")

    assert result.prompt == "candidate"
    assert architect.design_prompt.call_count == EXPECTED_RETRY_COUNT
    assert pilot.test_candidate.call_count == EXPECTED_RETRY_COUNT
    assert judge.evaluate.call_count == EXPECTED_RETRY_COUNT
    assert scoring.calculate_score.call_count == EXPECTED_RETRY_COUNT


@pytest.mark.asyncio
async def test_pipeline_early_stopping(mock_roles, mock_model):
    historian, decompiler, architect, pilot, judge, scoring = mock_roles

    # Scores don't improve
    scoring.calculate_score.side_effect = [0.5, 0.4, 0.4]

    pipeline = PromptCompilerPipeline(
        historian=historian,
        decompiler=decompiler,
        architect=architect,
        pilot=pilot,
        judge=judge,
        scoring_algorithm=scoring,
        score_threshold=0.9,
        max_retries=5,
        early_stop_patience=2,
    )

    await pipeline.run(raw_prompt="raw", source_model="gpt-4", target_model="gemini-pro")

    # Should stop after 3 attempts (0 (best), 1 (no improve), 2 (no improve -> stop))
    # Wait, logic is:
    # 1. score 0.5. best=0.5. patience=0.
    # 2. score 0.4. best=0.5. patience=1.
    # 3. score 0.4. best=0.5. patience=2. -> break

    assert architect.design_prompt.call_count == EXPECTED_EARLY_STOP_COUNT


@pytest.mark.asyncio
async def test_pipeline_max_retries_reached(mock_roles, mock_model):
    historian, decompiler, architect, pilot, judge, scoring = mock_roles

    scoring.calculate_score.return_value = 0.5

    pipeline = PromptCompilerPipeline(
        historian=historian,
        decompiler=decompiler,
        architect=architect,
        pilot=pilot,
        judge=judge,
        scoring_algorithm=scoring,
        score_threshold=0.9,
        max_retries=1,
    )

    result = await pipeline.run(raw_prompt="raw", source_model="gpt-4", target_model="gemini-pro")

    assert result.prompt == "candidate"
    assert architect.design_prompt.call_count == EXPECTED_RETRY_COUNT  # Initial + 1 retry


@pytest.mark.asyncio
async def test_pipeline_failure(mock_roles):
    historian, decompiler, architect, pilot, judge, scoring = mock_roles

    historian.establish_baseline.side_effect = Exception("Boom")

    pipeline = PromptCompilerPipeline(
        historian=historian,
        decompiler=decompiler,
        architect=architect,
        pilot=pilot,
        judge=judge,
        scoring_algorithm=scoring,
    )

    with pytest.raises(Exception, match="Boom"):
        await pipeline.run(raw_prompt="raw", source_model="gpt-4", target_model="gemini-pro")


@pytest.mark.asyncio
async def test_compile_pipeline_entry_point():
    with patch("prompt_compiler.core.pipeline.PromptCompilerPipeline") as MockPipeline:
        mock_instance = MockPipeline.return_value
        mock_instance.run = AsyncMock(return_value="result")

        result = await compile_pipeline("raw", "src", "tgt")

        assert result == "result"  # type: ignore
        mock_instance.run.assert_called_once()
