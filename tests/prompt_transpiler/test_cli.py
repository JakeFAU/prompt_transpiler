from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from prompt_transpiler.cli import _update_role_settings, main
from prompt_transpiler.config import settings
from prompt_transpiler.dto.models import Model, ModelProviderType, PromptStyle, Provider
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    # The output depends on how click is invoked, often defaults to 'main' if not installed
    # or specified. We just check for version number.
    assert "version 0.1.0" in result.output


def test_cli_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Prompt Transpiler CLI" in result.output
    assert "--max-retries" in result.output


def test_cli_no_args_shows_help(runner: CliRunner) -> None:
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output


@patch("prompt_transpiler.cli.transpile_pipeline", new_callable=AsyncMock)
def test_cli_with_text_input(mock_pipeline: AsyncMock, runner: CliRunner) -> None:
    # Setup mock to return a result
    mock_result = CandidatePrompt(
        prompt="Optimized Prompt",
        model=Model(
            provider=Provider(provider="gemini", provider_type=ModelProviderType.API),
            model_name="gemini-2.5",
            supports_system_messages=True,
            context_window_size=1000000,
            prompt_style=PromptStyle.MARKDOWN,
            supports_json_mode=True,
            prompting_tips="Be concise.",
        ),
    )
    mock_pipeline.return_value = mock_result

    result = runner.invoke(
        main,
        [
            "Simple prompt",
            "-s",
            "gpt-4",
            "-t",
            "gemini-2.5",
            "--source-provider",
            "openai",
            "--target-provider",
            "gemini",
            "--diff-provider",
            "openai",
            "--diff-model",
            "gpt-4o-mini",
        ],
    )

    if result.exit_code != 0:
        print(result.output)

    assert result.exit_code == 0
    # Check if transpile_pipeline was called with correct args
    mock_pipeline.assert_called_once()
    args, _kwargs = mock_pipeline.call_args
    assert args[0] == "Simple prompt"
    assert args[1] == "gpt-4"
    assert args[2] == "gemini-2.5"


@patch("prompt_transpiler.cli.transpile_pipeline", new_callable=AsyncMock)
def test_cli_output_file(mock_pipeline: AsyncMock, runner: CliRunner, tmp_path) -> None:
    mock_result = CandidatePrompt(
        prompt="Optimized Prompt Content",
        model=Model(
            provider=Provider(provider="gemini", provider_type=ModelProviderType.API),
            model_name="gemini-2.5-flash",
            supports_system_messages=True,
            context_window_size=1000000,
            prompt_style=PromptStyle.MARKDOWN,
            supports_json_mode=True,
            prompting_tips="Be concise.",
        ),
    )
    mock_pipeline.return_value = mock_result

    output_file = tmp_path / "output.txt"

    result = runner.invoke(
        main,
        [
            "Input prompt",
            "-o",
            str(output_file),
            "--source-provider",
            "openai",
            "--target-provider",
            "gemini",
            "--diff-provider",
            "openai",
            "--diff-model",
            "gpt-4o-mini",
        ],
    )

    if result.exit_code != 0:
        print(result.output)

    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "Optimized Prompt Content"


@patch("prompt_transpiler.cli.transpile_pipeline", new_callable=AsyncMock)
def test_cli_report_json(mock_pipeline: AsyncMock, runner: CliRunner, tmp_path) -> None:
    mock_result = CandidatePrompt(
        prompt="Optimized Prompt Content",
        model=Model(
            provider=Provider(provider="gemini", provider_type=ModelProviderType.API),
            model_name="gemini-2.5-flash",
            supports_system_messages=True,
            context_window_size=1000000,
            prompt_style=PromptStyle.MARKDOWN,
            supports_json_mode=True,
            prompting_tips="Be concise.",
        ),
    )
    mock_result.primary_intent_score = 0.91
    mock_result.tone_voice_score = 0.83
    mock_result.constraint_scores = {"json": 0.95}
    mock_result.run_metadata = {"scoring_algo": "weighted"}
    mock_pipeline.return_value = mock_result

    report_file = tmp_path / "report.json"

    result = runner.invoke(
        main,
        [
            "Input prompt",
            "--report-json",
            str(report_file),
            "--show-scores",
            "--source-provider",
            "openai",
            "--target-provider",
            "gemini",
            "--diff-provider",
            "openai",
            "--diff-model",
            "gpt-4o-mini",
        ],
    )

    assert result.exit_code == 0
    assert report_file.exists()
    report = report_file.read_text(encoding="utf-8")
    assert '"transpiled_prompt": "Optimized Prompt Content"' in report
    assert '"attempts": []' in report
    assert "Score Summary" in result.output


def test_cli_updates_settings(runner: CliRunner) -> None:
    # Original values
    orig_arch_prov = settings.get("roles.architect.provider")

    try:
        _update_role_settings(
            architect_provider="test_provider",
            architect_model="test_model",
            decompiler_provider=None,
            decompiler_model=None,
            diff_provider="diff_test_provider",
            diff_model="diff_test_model",
            judge_provider=None,
            judge_model=None,
        )

        assert settings.roles.architect.provider == "test_provider"
        assert settings.roles.architect.model == "test_model"
        assert settings.roles.diff.provider == "diff_test_provider"
        assert settings.roles.diff.model == "diff_test_model"

    finally:
        # cleanup/restore if needed, though settings might be process global
        if orig_arch_prov:
            settings.set("roles.architect.provider", orig_arch_prov)
