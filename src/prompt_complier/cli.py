import asyncio
import logging
from pathlib import Path

import click
from dynaconf import settings

from prompt_complier.core.pipeline import compile_pipeline
from prompt_complier.utils.logging import get_logger

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
    if architect_provider:
        settings.set("ARCHITECT_PROVIDER", architect_provider)
    if architect_model:
        settings.set("ARCHITECT_MODEL", architect_model)
    if decompiler_provider:
        settings.set("DECOMPILER_PROVIDER", decompiler_provider)
    if decompiler_model:
        settings.set("DECOMPILER_MODEL", decompiler_model)
    if judge_provider:
        settings.set("JUDGE_PROVIDER", judge_provider)
    if judge_model:
        settings.set("JUDGE_MODEL", judge_model)


def _load_prompt(prompt_input: str, logger: logging.Logger) -> str | None:
    """Load prompt from input string or file."""
    prompt_path = Path(prompt_input)
    if prompt_path.is_file():
        try:
            raw_text = prompt_path.read_text(encoding="utf-8")
            logger.info("Loaded prompt from file", path=str(prompt_path))  # type: ignore[call-arg]
            return raw_text
        except Exception as e:
            logger.error("Failed to read prompt file", path=str(prompt_path), error=str(e))  # type: ignore[call-arg]
            click.echo(f"Error reading file: {e}", err=True)
            return None
    else:
        return prompt_input


@click.command()
@click.argument("prompt_input", required=False)
@click.option(
    "--source",
    "-s",
    default="gpt-4o-mini",
    help="Source model name (e.g., gpt-4o-mini)",
)
@click.option(
    "--target",
    "-t",
    default="gemini-1.5-flash",
    help="Target model name (e.g., gemini-1.5-flash)",
)
@click.option(
    "--source-provider",
    default="openai",
    help="Provider for source model (default: openai)",
)
@click.option(
    "--target-provider",
    default="gemini",
    help="Provider for target model (default: gemini)",
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
    "--log-level",
    default="INFO",
    help="Log level (DEBUG, INFO, WARNING, ERROR)",
)
def main(  # noqa: PLR0913
    source: str,
    target: str,
    source_provider: str,
    target_provider: str,
    env: str,
    log_level: str,
    prompt_input: str | None = None,
    architect_provider: str | None = None,
    architect_model: str | None = None,
    decompiler_provider: str | None = None,
    decompiler_model: str | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> None:
    """
    Prompt Compiler CLI.

    PROMPT_INPUT: The raw prompt text or a path to a text file containing the prompt.
    """
    settings.setenv(env)
    settings.update({"LOG_LEVEL": log_level})

    _update_role_settings(
        architect_provider,
        architect_model,
        decompiler_provider,
        decompiler_model,
        judge_provider,
        judge_model,
    )

    if not prompt_input:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        return

    raw_text = _load_prompt(prompt_input, logger)  # type: ignore[arg-type]
    if raw_text is None:
        return

    logger.info("Starting compilation", source=source, target=target, env=env)

    async def run_async() -> None:
        try:
            result = await compile_pipeline(
                raw_text,
                source,
                target,
                source_provider=source_provider,
                target_provider=target_provider,
            )
            click.echo("\n--- Compiled Prompt ---\n")
            click.echo(result.prompt)
            click.echo("\n-----------------------\n")
        except Exception as e:
            logger.error("Compilation failed", error=str(e))
            click.echo(f"Error during compilation: {e}", err=True)
            if log_level.upper() == "DEBUG":
                raise

    asyncio.run(run_async())


if __name__ == "__main__":
    main()
