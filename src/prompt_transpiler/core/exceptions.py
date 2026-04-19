"""
Custom exceptions for the Prompt Transpiler core logic.
"""


class PromptTranspilerError(Exception):
    """Base exception for all prompt transpiler errors."""

    pass


class ProviderError(PromptTranspilerError):
    """Raised when an LLM provider fails."""

    pass


class DecompilationError(PromptTranspilerError):
    """Raised when the Decompiler fails to extract a valid IR."""

    pass


class ArchitectureError(PromptTranspilerError):
    """Raised when the Architect fails to generate a candidate."""

    pass


class EvaluationError(PromptTranspilerError):
    """Raised when the Judge fails to evaluate."""

    pass


DetranspilationError = DecompilationError
PromptCompilerError = PromptTranspilerError
