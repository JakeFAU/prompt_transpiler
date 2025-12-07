from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from prompt_compiler.cli import _update_role_settings, main
from prompt_compiler.config import settings


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
    assert "Prompt Compiler CLI" in result.output
    assert "--max-retries" in result.output


def test_cli_no_args_shows_help(runner: CliRunner) -> None:
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output


@patch("prompt_compiler.cli.compile_pipeline", new_callable=AsyncMock)
def test_cli_with_text_input(mock_pipeline: AsyncMock, runner: CliRunner) -> None:
    # Setup mock to return a result
    mock_result = AsyncMock()
    mock_result.prompt = "Optimized Prompt"
    mock_pipeline.return_value = mock_result

    result = runner.invoke(main, ["Simple prompt", "-s", "gpt-4", "-t", "gemini-2.5"])

    if result.exit_code != 0:
        print(result.output)

    assert result.exit_code == 0
    # Check if compile_pipeline was called with correct args
    mock_pipeline.assert_called_once()
    args, _kwargs = mock_pipeline.call_args
    assert args[0] == "Simple prompt"
    assert args[1] == "gpt-4"
    assert args[2] == "gemini-2.5"


@patch("prompt_compiler.cli.compile_pipeline", new_callable=AsyncMock)
def test_cli_output_file(mock_pipeline: AsyncMock, runner: CliRunner, tmp_path) -> None:
    mock_result = AsyncMock()
    mock_result.prompt = "Optimized Prompt Content"
    mock_pipeline.return_value = mock_result

    output_file = tmp_path / "output.txt"

    result = runner.invoke(main, ["Input prompt", "-o", str(output_file)])

    if result.exit_code != 0:
        print(result.output)

    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "Optimized Prompt Content"


def test_cli_updates_settings(runner: CliRunner) -> None:
    # Original values
    orig_arch_prov = settings.get("roles.architect.provider")

    try:
        _update_role_settings(
            architect_provider="test_provider",
            architect_model="test_model",
            decompiler_provider=None,
            decompiler_model=None,
            judge_provider=None,
            judge_model=None,
        )

        assert settings.roles.architect.provider == "test_provider"
        assert settings.roles.architect.model == "test_model"

    finally:
        # cleanup/restore if needed, though settings might be process global
        if orig_arch_prov:
            settings.set("roles.architect.provider", orig_arch_prov)
