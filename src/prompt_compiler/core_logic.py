"""
Legacy entry point. Redirects to `prompt_compiler.core.pipeline`.
"""

from prompt_compiler.core.pipeline import compile_pipeline
from prompt_compiler.llm.prompts.prompt_objects import (
    CandidatePrompt,
    OriginalPrompt,
    ScoringAlgorithm,
)

__all__ = ["CandidatePrompt", "OriginalPrompt", "ScoringAlgorithm", "compile_pipeline"]
