import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from prompt_transpiler.cli import main
from prompt_transpiler.dto.models import (
    Message,
    Model,
    ModelProviderType,
    PromptPayload,
    PromptStyle,
    Provider,
)
from prompt_transpiler.llm.prompts.prompt_objects import CandidatePrompt


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@patch("prompt_transpiler.cli.transpile_pipeline", new_callable=AsyncMock)
def test_cli_json_input(mock_pipeline: AsyncMock, runner: CliRunner, tmp_path: Path) -> None:
    # Setup mock to return a result
    mock_result = CandidatePrompt(
        payload=PromptPayload(messages=[Message(role="user", content="Optimized Prompt")]),
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

    # Create a JSON input file
    payload_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    }
    json_file = tmp_path / "input.json"
    json_file.write_text(json.dumps(payload_data))

    result = runner.invoke(
        main,
        [
            str(json_file),
            "-s",
            "gpt-4",
            "-t",
            "gemini-2.5",
            "--source-provider",
            "openai",
            "--target-provider",
            "gemini",
        ],
    )

    assert result.exit_code == 0
    # Check if transpile_pipeline was called with correct args
    mock_pipeline.assert_called_once()
    args, _kwargs = mock_pipeline.call_args

    # We expect the pipeline to receive either the PromptPayload object or the serialized version
    # depending on how we implement it. Task 6 says:
    # "attempt to parse it as a PromptPayload... If it's a normal string, continue to wrap
    # it in a single user message."
    # If we update the pipeline to handle PromptPayload, that's better.

    # For now, let's see how we want to pass it.
    # If I pass it as PromptPayload, it's more robust.
    assert isinstance(args[0], PromptPayload)
    expected_msg_count = 2
    assert len(args[0].messages) == expected_msg_count
    assert args[0].messages[0].role == "system"


@patch("prompt_transpiler.cli.transpile_pipeline", new_callable=AsyncMock)
def test_cli_json_output(mock_pipeline: AsyncMock, runner: CliRunner) -> None:
    # Setup mock to return a result with specific payload
    mock_result = CandidatePrompt(
        payload=PromptPayload(
            messages=[
                Message(role="system", content="System instruction"),
                Message(role="user", content="User message"),
            ]
        ),
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
            "--output-json",
            "-s",
            "gpt-4",
            "-t",
            "gemini-2.5",
            "--source-provider",
            "openai",
            "--target-provider",
            "gemini",
        ],
    )

    assert result.exit_code == 0
    # Output should be valid JSON and contain the messages
    # stdout should be clean now as we moved other info to stderr
    output_json = json.loads(result.stdout)
    # Actually, if --output-json is set, we might want to suppress other output or handle it
    # carefully.
    # Task says: "If --output-json is provided, print the serialized payload."

    assert "messages" in output_json
    expected_msg_count = 2
    assert len(output_json["messages"]) == expected_msg_count
    assert output_json["messages"][0]["role"] == "system"
    assert output_json["messages"][0]["content"] == "System instruction"
