"""
Custom exceptions for the Prompt Compiler core logic.
"""


class PromptCompilerError(Exception):
    """Base exception for all prompt compiler errors."""

    pass


class ProviderError(PromptCompilerError):
    """Raised when an LLM provider fails."""

    pass


class DecompilationError(PromptCompilerError):
    """Raised when the Decompiler fails to extract a valid IR."""

    pass


class ArchitectureError(PromptCompilerError):
    """Raised when the Architect fails to generate a candidate."""

    pass


class EvaluationError(PromptCompilerError):
    """Raised when the Judge fails to evaluate."""

    pass
