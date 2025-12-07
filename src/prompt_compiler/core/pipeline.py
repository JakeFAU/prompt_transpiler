from typing import Any

from attrs import define, field

from prompt_compiler.config import settings
from prompt_compiler.core.interfaces import (
    IArchitect,
    IDecompiler,
    IHistorian,
    IJudge,
    IPilot,
)
from prompt_compiler.core.roles.architect import GPTArchitect
from prompt_compiler.core.roles.decompiler import GeminiDecompiler
from prompt_compiler.core.roles.historian import DefaultHistorian
from prompt_compiler.core.roles.pilot import DefaultPilot
from prompt_compiler.core.scoring import LLMAdjudicator, WeightedScoreAlgorithm
from prompt_compiler.dto.models import Model, ModelProviderType, PromptStyle, Provider
from prompt_compiler.llm.prompts.prompt_objects import (
    CandidatePrompt,
    OriginalPrompt,
    ScoringAlgorithm,
)
from prompt_compiler.utils.logging import get_logger
from prompt_compiler.utils.telemetry import telemetry

logger = get_logger(__name__)

# --- Helper for Dummy Models ---


def create_dummy_model(model_name: str, provider_name: str) -> Model:
    """Helper to create a Model object since we don't have a database yet."""
    p_type = ModelProviderType.API
    normalized_provider = provider_name.lower().strip()
    if "huggingface" in normalized_provider:
        p_type = ModelProviderType.HUGGINGFACE

    style = PromptStyle.MARKDOWN
    if "claude" in model_name.lower():
        style = PromptStyle.XML

    return Model(
        provider=Provider(
            provider=normalized_provider,
            provider_type=p_type,
        ),
        model_name=model_name,
        supports_system_messages=True,
        context_window_size=8192,
        prompt_style=style,
        supports_json_mode=True,
        prompting_tips="Be concise.",
    )


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


@define
class PromptCompilerPipeline:
    """
    The main orchestration engine for the Prompt Compiler.

    Uses Dependency Injection for all roles to allow swapping strategies.
    """

    historian: IHistorian = field(factory=DefaultHistorian)
    decompiler: IDecompiler = field(factory=_default_decompiler)
    architect: IArchitect = field(factory=_default_architect)
    pilot: IPilot = field(factory=DefaultPilot)
    judge: IJudge = field(factory=_default_judge)

    # Injectable Scoring Strategy
    scoring_algorithm: ScoringAlgorithm = field(factory=WeightedScoreAlgorithm)

    score_threshold: float = field(default=settings.COMPILER.SCORE_THRESHOLD)
    max_retries: int = field(default=settings.COMPILER.MAX_RETRIES)
    early_stop_patience: int = field(default=settings.COMPILER.EARLY_STOP_PATIENCE)

    # Telemetry Metrics
    _run_counter: Any = field(init=False)
    _success_counter: Any = field(init=False)
    _failure_counter: Any = field(init=False)
    _retry_counter: Any = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Initialize telemetry metrics."""
        self._run_counter = telemetry.get_counter(
            "compiler.pipeline.runs", "Number of pipeline executions"
        )
        self._success_counter = telemetry.get_counter(
            "compiler.pipeline.success", "Number of successful pipeline completions"
        )
        self._failure_counter = telemetry.get_counter(
            "compiler.pipeline.failures", "Number of failed pipeline executions"
        )
        self._retry_counter = telemetry.get_counter(
            "compiler.pipeline.retries", "Number of optimization retries"
        )

    @telemetry.instrument(name="pipeline.compile")
    async def run(  # noqa: PLR0915, PLR0913
        self,
        raw_prompt: str,
        source_model: str,
        target_model: str,
        max_retries: int | None = None,
        source_provider: str = "openai",
        target_provider: str = "openai",
    ) -> CandidatePrompt:
        """
        Execute the prompt compilation pipeline.

        Args:
            raw_prompt: The original prompt text to be converted.
            source_model: Name of the model the original prompt was designed for (e.g., 'gpt-4').
            target_model: Name of the model to optimize for (e.g., 'gemini-1.5-pro').
            max_retries: Optional override for maximum optimization attempts.
            source_provider: Provider for source model (default: openai).
            target_provider: Provider for target model (default: openai).
        """
        if max_retries is None:
            max_retries = self.max_retries

        self._run_counter.add(1, {"source": source_model, "target": target_model})

        logger.info(
            "Starting compilation pipeline",
            source=source_model,
            target=target_model,
            max_retries=max_retries,
        )

        try:
            # 1. Setup Models
            # Now we use passed providers
            src_model_obj = create_dummy_model(source_model, source_provider)
            tgt_model_obj = create_dummy_model(target_model, target_provider)

            original = OriginalPrompt(prompt=raw_prompt, model=src_model_obj)

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
                if final_score > best_score:
                    best_score = final_score
                    best_candidate = candidate
                    patience_counter = 0  # Reset patience
                else:
                    patience_counter += 1

                # Success check
                if final_score >= self.score_threshold:
                    logger.info("Threshold met!", score=final_score)
                    self._success_counter.add(1)
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
                self._success_counter.add(1, {"status": "max_retries_best_effort"})
                return best_candidate

            # Fallback (should rarely happen unless first attempt failed hard)
            if candidate:
                self._success_counter.add(1, {"status": "fallback"})
                return candidate

            raise RuntimeError("Pipeline failed to generate any candidate prompt")

        except Exception as e:
            logger.error("Pipeline failed execution", error=str(e))
            self._failure_counter.add(1)
            raise
        return candidate


async def compile_pipeline(
    raw_text: str,
    source_model_name: str,
    target_model_name: str,
    source_provider: str = "openai",
    target_provider: str = "openai",
) -> CandidatePrompt:
    """Entry point for the CLI or API."""
    pipeline = PromptCompilerPipeline()
    return await pipeline.run(
        raw_text,
        source_model_name,
        target_model_name,
        source_provider=source_provider,
        target_provider=target_provider,
    )
