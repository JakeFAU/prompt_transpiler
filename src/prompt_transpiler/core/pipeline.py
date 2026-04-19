"""Prompt transpilation orchestration pipeline."""

from typing import Any

from attrs import define, field

from prompt_transpiler.config import settings
from prompt_transpiler.core.interfaces import (
    IArchitect,
    IDecompiler,
    IDiffAgent,
    IHistorian,
    IJudge,
    IPilot,
)
from prompt_transpiler.core.registry import ModelRegistry
from prompt_transpiler.core.roles.architect import GPTArchitect
from prompt_transpiler.core.roles.decompiler import GeminiDecompiler
from prompt_transpiler.core.roles.diff import SemanticDiffAgent
from prompt_transpiler.core.roles.historian import DefaultHistorian
from prompt_transpiler.core.roles.pilot import DefaultPilot
from prompt_transpiler.core.scoring import (
    LLMAdjudicator,
    PairwisePreferenceAlgorithm,
    get_scoring_algorithm,
)
from prompt_transpiler.dto.models import Message, PromptPayload
from prompt_transpiler.llm.prompts.prompt_objects import (
    CandidatePrompt,
    CompilationAttempt,
    OriginalPrompt,
    ScoringAlgorithm,
)
from prompt_transpiler.utils.logging import get_logger
from prompt_transpiler.utils.telemetry import telemetry

logger = get_logger(__name__)


def _default_architect() -> IArchitect:
    return GPTArchitect(
        provider_name=settings.roles.architect.provider,
        model_name=settings.roles.architect.model,
    )


def _default_decompiler() -> IDecompiler:
    return GeminiDecompiler(
        provider_name=settings.roles.decompiler.provider,
        model_name=settings.roles.decompiler.model,
    )


def _default_judge() -> IJudge:
    return LLMAdjudicator(
        provider_name=settings.roles.judge.provider,
        model_name=settings.roles.judge.model,
    )


def _default_diff_agent() -> IDiffAgent:
    return SemanticDiffAgent(
        provider_name=settings.roles.diff.provider,
        model_name=settings.roles.diff.model,
    )


@define
class PromptTranspilerPipeline:
    """
    The main orchestration engine for the Prompt Transpiler.

    Uses Dependency Injection for all roles to allow swapping strategies.
    """

    historian: IHistorian = field(factory=DefaultHistorian)
    decompiler: IDecompiler = field(factory=_default_decompiler)
    architect: IArchitect = field(factory=_default_architect)
    pilot: IPilot = field(factory=DefaultPilot)
    judge: IJudge = field(factory=_default_judge)
    diff_agent: IDiffAgent = field(factory=_default_diff_agent)

    # Model Registry
    registry: ModelRegistry = field(factory=ModelRegistry)

    # Injectable Scoring Strategy
    scoring_algorithm: ScoringAlgorithm = field(factory=PairwisePreferenceAlgorithm)
    scoring_algorithm_name: str = field(default="pairwise")

    score_threshold: float = field(default=settings.TRANSPILER.SCORE_THRESHOLD)
    max_retries: int = field(default=settings.TRANSPILER.MAX_RETRIES)
    early_stop_patience: int = field(default=settings.TRANSPILER.EARLY_STOP_PATIENCE)

    # Telemetry Metrics
    _run_counter: Any = field(init=False)
    _success_counter: Any = field(init=False)
    _failure_counter: Any = field(init=False)
    _retry_counter: Any = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Initialize telemetry metrics."""
        self._run_counter = telemetry.get_counter(
            "transpiler.pipeline.runs", "Number of pipeline executions"
        )
        self._success_counter = telemetry.get_counter(
            "transpiler.pipeline.success", "Number of successful pipeline completions"
        )
        self._failure_counter = telemetry.get_counter(
            "transpiler.pipeline.failures", "Number of failed pipeline executions"
        )
        self._retry_counter = telemetry.get_counter(
            "transpiler.pipeline.retries", "Number of optimization retries"
        )

        # Inject strategy into Judge if supported (for Telemetry)
        if isinstance(self.judge, LLMAdjudicator):
            self.judge.scoring_algorithm = self.scoring_algorithm
            self.judge.score_threshold = self.score_threshold

    async def _annotate_diff(
        self, candidate: CandidatePrompt | None, original: OriginalPrompt
    ) -> CandidatePrompt | None:
        """
        Run the diff agent to summarize how the candidate differs from the original prompt.
        Failures are swallowed to avoid blocking transpilation.
        """
        if candidate is None:
            return None

        try:
            candidate.diff_summary = await self.diff_agent.summarize_diff(original, candidate)
        except Exception as exc:
            logger.error("Diff agent failed", error=str(exc))

        return candidate

    def _attach_run_metadata(self, candidate: CandidatePrompt, run_context: dict[str, Any]) -> None:
        """Attach machine-readable run metadata to the candidate."""
        candidate.run_metadata = {
            "max_retries": run_context["max_retries"],
            "score_threshold": self.score_threshold,
            "scoring_algo": self.scoring_algorithm_name,
            "source_provider": run_context["source_provider"],
            "target_provider": run_context["target_provider"],
            "source_model": run_context["source_model"],
            "target_model": run_context["target_model"],
        }

    @telemetry.instrument(name="pipeline.transpile")
    async def run(  # noqa: PLR0915, PLR0913, PLR0912
        self,
        raw_prompt: str | PromptPayload,
        source_model: str,
        target_model: str,
        max_retries: int | None = None,
        source_provider: str = "openai",
        target_provider: str = "openai",
    ) -> CandidatePrompt:
        """
        Execute the prompt transpilation pipeline.

        Args:
            raw_prompt: The original prompt text or PromptPayload to be converted.
            source_model: Name of the model the original prompt was designed for (e.g., 'gpt-4').
            target_model: Name of the model to optimize for (e.g., 'gemini-2.5-pro').
            max_retries: Optional override for maximum optimization attempts.
            source_provider: Provider for source model (default: openai).
            target_provider: Provider for target model (default: openai).
        """
        if max_retries is None:
            max_retries = self.max_retries

        self._run_counter.add(1, {"source": source_model, "target": target_model})

        logger.info(
            "Starting transpilation pipeline",
            source=source_model,
            target=target_model,
            max_retries=max_retries,
        )

        try:
            # 1. Setup Models
            # Now we use passed providers and the registry
            src_model_obj = self.registry.get_model(source_model, source_provider)
            tgt_model_obj = self.registry.get_model(target_model, target_provider)

            if isinstance(raw_prompt, str):
                payload = PromptPayload(messages=[Message(role="user", content=raw_prompt)])
            else:
                payload = raw_prompt

            original = OriginalPrompt(
                payload=payload,
                model=src_model_obj,
            )

            # 2. Establish Baseline
            logger.debug("Stage 1: Establishing Baseline")
            original = await self.historian.establish_baseline(original)

            # 3. Decompile to IR
            logger.debug("Stage 2: Decompiling to IR")
            ir = await self.decompiler.decompile(original, tgt_model_obj)

            # 4. Architect & Optimize Loop
            logger.debug("Stage 3: Entering Optimization Loop")
            best_candidate: CandidatePrompt | None = None
            candidate: CandidatePrompt | None = None
            best_score = -1.0
            patience_counter = 0
            feedback: str | None = None
            attempts: list[CompilationAttempt] = []
            run_context = {
                "source_model": source_model,
                "target_model": target_model,
                "source_provider": source_provider,
                "target_provider": target_provider,
                "max_retries": max_retries,
            }

            for attempt in range(max_retries + 1):
                logger.info(f"Optimization loop: Attempt {attempt + 1}/{max_retries + 1}")
                if attempt > 0:
                    self._retry_counter.add(1)

                # Generate (Architect)
                candidate = await self.architect.design_prompt(ir, tgt_model_obj, feedback=feedback)

                # Test (Pilot)
                candidate = await self.pilot.test_candidate(candidate)

                # Evaluate (Judge)
                await self.judge.evaluate(candidate, original)

                # Calculate Score (Algorithm)
                final_score = candidate.total_score(self.scoring_algorithm, original)

                logger.info("Candidate scored", score=final_score)

                # Update Best
                is_new_best = final_score > best_score
                if final_score > best_score:
                    best_score = final_score
                    best_candidate = candidate
                    patience_counter = 0  # Reset patience
                else:
                    patience_counter += 1

                accepted = final_score >= self.score_threshold
                attempts.append(
                    CompilationAttempt(
                        attempt=attempt + 1,
                        final_score=final_score,
                        primary_intent_score=candidate.primary_intent_score,
                        tone_voice_score=candidate.tone_voice_score,
                        constraint_scores=candidate.constraint_scores,
                        primary_intent_verdict=candidate.primary_intent_verdict,
                        tone_voice_verdict=candidate.tone_voice_verdict,
                        constraint_verdicts=candidate.constraint_verdicts,
                        primary_intent_confidence=candidate.primary_intent_confidence,
                        tone_voice_confidence=candidate.tone_voice_confidence,
                        constraint_confidences=candidate.constraint_confidences,
                        feedback=candidate.feedback,
                        accepted=accepted,
                        new_best=is_new_best,
                    )
                )
                candidate.attempt_history = list(attempts)
                self._attach_run_metadata(candidate, run_context)

                # Success check
                if accepted:
                    logger.info("Threshold met!", score=final_score)
                    self._success_counter.add(1)
                    await self._annotate_diff(candidate, original)
                    return candidate

                # Early Stopping check
                if patience_counter >= self.early_stop_patience:
                    logger.warning("Early stopping triggered: No improvement.")
                    break

                # Prepare feedback for next loop
                feedback = candidate.feedback
                if not feedback:
                    logger.warning("Judge provided no feedback for optimization.")

            logger.warning("Max retries reached. Returning best candidate.", best_score=best_score)
            if best_candidate:
                best_candidate.attempt_history = list(attempts)
                self._attach_run_metadata(best_candidate, run_context)
                self._success_counter.add(1, {"status": "max_retries_best_effort"})
                await self._annotate_diff(best_candidate, original)
                return best_candidate

            # Fallback (should rarely happen unless first attempt failed hard)
            if candidate:
                candidate.attempt_history = list(attempts)
                self._attach_run_metadata(candidate, run_context)
                self._success_counter.add(1, {"status": "fallback"})
                await self._annotate_diff(candidate, original)
                return candidate

            raise RuntimeError("Pipeline failed to generate any candidate prompt")

        except Exception as e:
            logger.error("Pipeline failed execution", error=str(e))
            self._failure_counter.add(1)
            raise
        return candidate


async def transpile_pipeline(  # noqa: PLR0913
    raw_text: str | PromptPayload,
    source_model_name: str,
    target_model_name: str,
    source_provider: str = "openai",
    target_provider: str = "openai",
    max_retries: int | None = None,
    score_threshold: float | None = None,
    scoring_algo: str | None = None,
) -> CandidatePrompt:
    """Entry point for the CLI or API."""
    kwargs: dict[str, Any] = {}
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold

    # Resolve scoring algorithm
    # 1. CLI/Arg override
    # 2. Settings
    # 3. Default "weighted"
    algo_name = scoring_algo or settings.get("transpiler.scoring_algorithm", "weighted")
    algorithm = get_scoring_algorithm(algo_name)
    kwargs["scoring_algorithm"] = algorithm
    kwargs["scoring_algorithm_name"] = algo_name

    pipeline = PromptTranspilerPipeline(**kwargs)
    return await pipeline.run(
        raw_text,
        source_model_name,
        target_model_name,
        source_provider=source_provider,
        target_provider=target_provider,
        max_retries=max_retries,
    )


PromptCompilerPipeline = PromptTranspilerPipeline
compile_pipeline = transpile_pipeline
