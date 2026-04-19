"""
Legacy entry point. Redirects to `prompt_transpiler.core.pipeline`.
"""

from prompt_transpiler.core.pipeline import transpile_pipeline
from prompt_transpiler.llm.prompts.prompt_objects import (
    CandidatePrompt,
    OriginalPrompt,
    ScoringAlgorithm,
)

__all__ = ["CandidatePrompt", "OriginalPrompt", "ScoringAlgorithm", "transpile_pipeline"]
