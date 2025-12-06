"""
Legacy entry point. Redirects to `prompt_complier.core.pipeline`.
"""

from prompt_complier.core.pipeline import compile_pipeline
from prompt_complier.llm.prompts.prompt_objects import (
    CandidatePrompt,
    OriginalPrompt,
    ScoringAlgorithm,
)

__all__ = ["CandidatePrompt", "OriginalPrompt", "ScoringAlgorithm", "compile_pipeline"]
