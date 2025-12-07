import asyncio
import logging
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any

import click

from prompt_compiler.config import settings
from prompt_compiler.core.pipeline import compile_pipeline
from prompt_compiler.utils.logging import get_logger


def _get_version() -> str:
    """Get the package version, falling back to VERSION.txt if not installed."""
    try:
        return get_version("prompt-complier")
    except PackageNotFoundError:
        # Fallback for when package is not installed (e.g., CI/CD, development)
        version_file = Path(__file__).parent.parent.parent / "VERSION.txt"
        if version_file.exists():
            return version_file.read_text().strip()
        return "0.0.0"


logger = get_logger(__name__)


def _update_role_settings(  # noqa: PLR0913
    architect_provider: str | None,
    architect_model: str | None,
    decompiler_provider: str | None,
    decompiler_model: str | None,
    judge_provider: str | None,
    judge_model: str | None,
) -> None:
    """Update settings for specific roles if provided."""
    updates: dict[str, Any] = {}

    roles = updates.setdefault("roles", {})

    if architect_provider:
        roles.setdefault("architect", {})["provider"] = architect_provider
    if architect_model:
        roles.setdefault("architect", {})["model"] = architect_model

    if decompiler_provider:
        roles.setdefault("decompiler", {})["provider"] = decompiler_provider
    if decompiler_model:
        roles.setdefault("decompiler", {})["model"] = decompiler_model

    if judge_provider:
        roles.setdefault("judge", {})["provider"] = judge_provider
    if judge_model:
        roles.setdefault("judge", {})["model"] = judge_model

    if updates:
        settings.update(updates, merge=True)


def _load_prompt(prompt_input: str, logger: logging.Logger) -> str | None:
    """Load prompt from input string or file."""
    # Check if input is a file path that exists
    prompt_path = Path(prompt_input)
    # We check is_file() but also that the input isn't just a generic string
    # that happens to match a relative path accidentally, though valid paths
    # are usually prioritized.
    if prompt_path.is_file():
        try:
            raw_text = prompt_path.read_text(encoding="utf-8")
            logger.info(
                "Loaded prompt from file",
                path=str(prompt_path),
            )  # type: ignore[call-arg]
            return raw_text
        except Exception as e:
            logger.error(
                "Failed to read prompt file",
                path=str(prompt_path),
                error=str(e),
            )  # type: ignore[call-arg]
            click.echo(f"Error reading file: {e}", err=True)
            return None
    else:
        # Assume it's the raw text
        return prompt_input


def _configure_logging(verbose: int, quiet: bool) -> str:
    """Configure logging level based on flags."""
    if quiet:
        return "ERROR"
    if verbose >= 1:
        return "DEBUG"
    return "INFO"


@click.command()
@click.argument("prompt_input", required=False)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True, path_type=Path),
    help="Output file path for the compiled prompt.",
)
@click.option(
    "--source",
    "-s",
    default="gpt-4o-mini",
    show_default=True,
    help="Source model name.",
)
@click.option(
    "--target",
    "-t",
    default="gemini-1.5-flash",
    show_default=True,
    help="Target model name.",
)
@click.option(
    "--source-provider",
    default="openai",
    show_default=True,
    help="Provider for source model.",
)
@click.option(
    "--target-provider",
    default="gemini",
    show_default=True,
    help="Provider for target model.",
)
@click.option(
    "--max-retries",
    type=int,
    help="Maximum number of optimization retries.",
)
@click.option(
    "--score-threshold",
    type=float,
    help="Score threshold to accept a prompt (0.0 to 1.0).",
)
@click.option(
    "--architect-provider",
    help="Provider for Architect agent",
)
@click.option(
    "--architect-model",
    help="Model for Architect agent",
)
@click.option(
    "--decompiler-provider",
    help="Provider for Decompiler agent",
)
@click.option(
    "--decompiler-model",
    help="Model for Decompiler agent",
)
@click.option(
    "--judge-provider",
    help="Provider for Judge agent",
)
@click.option(
    "--judge-model",
    help="Model for Judge agent",
)
@click.option(
    "--env",
    default="default",
    help="Dynaconf environment (e.g., development, production)",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (can be used multiple times).",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress output (only errors).",
)
@click.option(
    "--telemetry/--no-telemetry",
    default=True,
    show_default=True,
    help="Enable or disable OpenTelemetry.",
)
@click.version_option(version=_get_version())
def main(  # noqa: PLR0913
    prompt_input: str | None,
    output: Path | None,
    source: str,
    target: str,
    source_provider: str,
    target_provider: str,
    max_retries: int | None,
    score_threshold: float | None,
    architect_provider: str | None,
    architect_model: str | None,
    decompiler_provider: str | None,
    decompiler_model: str | None,
    judge_provider: str | None,
    judge_model: str | None,
    env: str,
    verbose: int,
    quiet: bool,
    telemetry: bool,
) -> None:
    """
    Prompt Compiler CLI.

    Compiles and optimizes a prompt from a source model to a target model.

    PROMPT_INPUT: The raw prompt text or a path to a text file containing the prompt.
    """
    # 1. Setup Configuration & Environment
    settings.setenv(env)

    # Configure Logging
    log_level = _configure_logging(verbose, quiet)
    settings.update({"LOG_LEVEL": log_level})

    # Configure Telemetry
    settings.set("USE_OPENTELEMETRY", telemetry)

    # 2. Update specific role settings
    # Note: Using double underscore for nested settings in Dynaconf if needed,
    # or specific keys depending on how config.py loads them.
    # Assuming config structure matches what's used in pipeline defaults.
    _update_role_settings(
        architect_provider,
        architect_model,
        decompiler_provider,
        decompiler_model,
        judge_provider,
        judge_model,
    )

    # 3. Validate Input
    if not prompt_input:
        # If no input provided, show help
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        sys.exit(0)

    raw_text = _load_prompt(prompt_input, logger)  # type: ignore[arg-type]
    if raw_text is None:
        sys.exit(1)

    if not quiet:
        click.echo(
            f"Compiling prompt from {source} ({source_provider}) to {target} ({target_provider})..."
        )

    logger.info("Starting compilation", source=source, target=target, env=env)

    # 4. Run Pipeline
    async def run_async() -> None:
        try:
            result = await compile_pipeline(
                raw_text,
                source,
                target,
                source_provider=source_provider,
                target_provider=target_provider,
                max_retries=max_retries,
                score_threshold=score_threshold,
            )

            # 5. Handle Output
            if output:
                output.write_text(result.prompt, encoding="utf-8")
                if not quiet:
                    click.echo(f"Compiled prompt saved to: {output}")
            else:
                if not quiet:
                    click.echo("\n--- Compiled Prompt ---\n")
                click.echo(result.prompt)
                if not quiet:
                    click.echo("\n-----------------------\n")

        except Exception as e:
            logger.error("Compilation failed", error=str(e))
            click.echo(f"Error during compilation: {e}", err=True)
            if verbose > 0:
                raise
            sys.exit(1)

    asyncio.run(run_async())


if __name__ == "__main__":
    main()
