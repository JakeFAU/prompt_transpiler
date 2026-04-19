"""Command-line interface for running the Prompt Transpiler pipeline."""

import asyncio
import json
import logging
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any

import click

from prompt_transpiler.config import settings
from prompt_transpiler.core.pipeline import transpile_pipeline
from prompt_transpiler.core.registry import ModelRegistry
from prompt_transpiler.reporting import build_transpile_report
from prompt_transpiler.utils.logging import get_logger
from prompt_transpiler.utils.token_collector import token_collector


def _get_version() -> str:
    """Get the package version, falling back to VERSION.txt if not installed."""
    try:
        return get_version("prompt-transpiler")
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
    diff_provider: str | None,
    diff_model: str | None,
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

    if diff_provider:
        roles.setdefault("diff", {})["provider"] = diff_provider
    if diff_model:
        roles.setdefault("diff", {})["model"] = diff_model

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
                path=str(prompt_path),  # pyright: ignore[reportCallIssue]
            )  # type: ignore[call-arg]
            return raw_text
        except Exception as e:
            logger.error(
                "Failed to read prompt file",
                path=str(prompt_path),  # pyright: ignore[reportCallIssue]
                error=str(e),  # pyright: ignore[reportCallIssue]
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


def _token_usage_snapshot() -> dict[str, dict[str, int]]:
    summary = token_collector.get_summary()
    return {
        model: {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        for model, usage in summary.items()
    }


def _token_usage_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    delta: dict[str, dict[str, int]] = {}
    for model, usage in after.items():
        prior = before.get(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        model_delta = {
            "prompt_tokens": max(0, usage["prompt_tokens"] - prior["prompt_tokens"]),
            "completion_tokens": max(0, usage["completion_tokens"] - prior["completion_tokens"]),
            "total_tokens": max(0, usage["total_tokens"] - prior["total_tokens"]),
        }
        if any(model_delta.values()):
            delta[model] = model_delta
    return delta


def _echo_score_summary(report: dict[str, Any]) -> None:
    scores = report["scores"]
    click.echo("\n--- Score Summary ---")
    click.echo(f"Final Score:         {scores['final_score']:.3f}")
    click.echo(f"Primary Intent:      {scores['primary_intent_score']}")
    click.echo(f"Tone / Voice:        {scores['tone_voice_score']}")
    constraint_scores = scores.get("constraint_scores") or {}
    if constraint_scores:
        click.echo("Constraint Scores:")
        for constraint, score in constraint_scores.items():
            click.echo(f"  - {constraint}: {score}")

    attempts = report.get("attempts") or []
    if attempts:
        click.echo("Attempts:")
        for attempt in attempts:
            status = "accepted" if attempt["accepted"] else "rejected"
            click.echo(f"  - #{attempt['attempt']}: score={attempt['final_score']:.3f} ({status})")
    click.echo("---------------------\n")


def _echo_token_summary(summary: dict[str, dict[str, int]]) -> None:
    click.echo("\n--- Token Usage Summary ---")
    total_cost_tokens = 0
    for model, usage in summary.items():
        click.echo(f"Model: {model}")
        click.echo(f"  Prompt Tokens:     {usage['prompt_tokens']}")
        click.echo(f"  Completion Tokens: {usage['completion_tokens']}")
        click.echo(f"  Total Tokens:      {usage['total_tokens']}")
        total_cost_tokens += usage["total_tokens"]
    click.echo(f"Grand Total Tokens:  {total_cost_tokens}")
    click.echo("---------------------------\n")


def _current_role_settings() -> dict[str, dict[str, str]]:
    return {
        "architect": {
            "provider": settings.roles.architect.provider,
            "model": settings.roles.architect.model,
        },
        "decompiler": {
            "provider": settings.roles.decompiler.provider,
            "model": settings.roles.decompiler.model,
        },
        "diff": {
            "provider": settings.roles.diff.provider,
            "model": settings.roles.diff.model,
        },
        "judge": {
            "provider": settings.roles.judge.provider,
            "model": settings.roles.judge.model,
        },
    }


def _build_cli_report(
    result: Any,
    *,
    registry: ModelRegistry,
    report_context: dict[str, Any],
    token_usage: dict[str, dict[str, int]],
) -> dict[str, Any]:
    source_model_obj = registry.get_model(
        report_context["source"], report_context["source_provider"]
    )
    target_model_obj = registry.get_model(
        report_context["target"], report_context["target_provider"]
    )
    final_score = getattr(result, "_cached_score", None)
    if final_score is None:
        final_score = 0.0
    result.run_metadata = {
        **getattr(result, "run_metadata", {}),
        "role_settings": _current_role_settings(),
        "requested": {
            "max_retries": report_context["max_retries"],
            "score_threshold": report_context["score_threshold"],
            "scoring_algo": report_context["scoring_algo"],
        },
    }
    return build_transpile_report(
        result,
        source_model=source_model_obj,
        target_model=target_model_obj,
        final_score=final_score,
        token_usage=token_usage,
    )


@click.command()
@click.argument("prompt_input", required=False)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True, path_type=Path),
    help="Output file path for the transpiled prompt.",
)
@click.option(
    "--report-json",
    type=click.Path(writable=True, path_type=Path),
    help="Write a machine-readable JSON report with scores, metadata, and attempts.",
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
    default="gemini-2.5-flash",
    show_default=True,
    help="Target model name.",
)
@click.option(
    "--source-provider",
    default=None,
    help="Provider for source model (auto-detected from registry if not specified).",
)
@click.option(
    "--target-provider",
    default=None,
    help="Provider for target model (auto-detected from registry if not specified).",
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
    "--scoring-algo",
    help="Scoring algorithm (weighted, geometric, penalty, dynamic).",
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
    "--diff-provider",
    help="Provider for Diff agent",
)
@click.option(
    "--diff-model",
    help="Model for Diff agent",
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
    "--show-scores/--hide-scores",
    default=False,
    show_default=True,
    help="Print a scoring summary for the selected candidate.",
)
@click.option(
    "--telemetry/--no-telemetry",
    default=True,
    show_default=True,
    help="Enable or disable OpenTelemetry.",
)
@click.version_option(version=_get_version())
def main(  # noqa: PLR0913, PLR0915
    prompt_input: str | None,
    output: Path | None,
    report_json: Path | None,
    source: str,
    target: str,
    source_provider: str,
    target_provider: str,
    max_retries: int | None,
    score_threshold: float | None,
    scoring_algo: str | None,
    architect_provider: str | None,
    architect_model: str | None,
    decompiler_provider: str | None,
    decompiler_model: str | None,
    diff_provider: str | None,
    diff_model: str | None,
    judge_provider: str | None,
    judge_model: str | None,
    env: str,
    verbose: int,
    quiet: bool,
    show_scores: bool,
    telemetry: bool,
) -> None:
    """
    Prompt Transpiler CLI.

    Transpiles and optimizes a prompt from a source model to a target model.

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
        diff_provider,
        diff_model,
        judge_provider,
        judge_model,
    )

    # 2b. Auto-detect providers from model registry if not specified
    registry = ModelRegistry()
    if source_provider is None:
        source_model = registry.get_model(source)
        source_provider = source_model.provider.provider
    if target_provider is None:
        target_model = registry.get_model(target)
        target_provider = target_model.provider.provider

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
        source_label = f"{source} ({source_provider})"
        target_label = f"{target} ({target_provider})"
        click.echo(f"Transpiling prompt from {source_label} to {target_label}...")

    logger.info("Starting transpilation", source=source, target=target, env=env)
    usage_before = _token_usage_snapshot()

    # 4. Run Pipeline
    async def run_async() -> None:
        try:
            result = await transpile_pipeline(
                raw_text,
                source,
                target,
                source_provider=source_provider,
                target_provider=target_provider,
                max_retries=max_retries,
                score_threshold=score_threshold,
                scoring_algo=scoring_algo,
            )
            usage_after = _token_usage_snapshot()
            token_usage = _token_usage_delta(usage_before, usage_after)
            report = _build_cli_report(
                result,
                registry=registry,
                report_context={
                    "source": source,
                    "target": target,
                    "source_provider": source_provider,
                    "target_provider": target_provider,
                    "max_retries": max_retries,
                    "score_threshold": score_threshold,
                    "scoring_algo": scoring_algo,
                },
                token_usage=token_usage,
            )

            if not quiet:
                _echo_token_summary(token_usage)

            if show_scores and not quiet:
                _echo_score_summary(report)

            if report_json:
                report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
                if not quiet:
                    click.echo(f"Report JSON saved to: {report_json}")

            # 5. Handle Output
            if output:
                output.write_text(result.prompt, encoding="utf-8")
                if not quiet:
                    click.echo(f"Transpiled prompt saved to: {output}")
            else:
                if not quiet:
                    click.echo("\n--- Transpiled Prompt ---\n")
                click.echo(result.prompt)
                if not quiet:
                    click.echo("\n-----------------------\n")

            # 6. Optional semantic diff explanation
            diff_summary = getattr(result, "diff_summary", None)
            if not quiet and diff_summary:
                click.echo("\n--- Semantic Diff ---\n")
                click.echo(diff_summary)
                click.echo("\n---------------------\n")

        except Exception as e:
            logger.error("Transpilation failed", error=str(e))
            click.echo(f"Error during transpilation: {e}", err=True)
            if verbose > 0:
                raise
            sys.exit(1)

    asyncio.run(run_async())


if __name__ == "__main__":
    main()
