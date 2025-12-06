"""
Core interfaces for the Prompt Compiler.

Defines the contracts for Roles and Strategies.
"""

from abc import ABC, abstractmethod

from prompt_complier.dto.models import IntermediateRepresentation, Model
from prompt_complier.llm.prompts.prompt_objects import CandidatePrompt, OriginalPrompt


class IHistorian(ABC):
    """Contract for the Historian Role (Baseline Runner)."""

    @abstractmethod
    async def establish_baseline(self, original_prompt: OriginalPrompt) -> OriginalPrompt:
        """Runs the original prompt to get a baseline response."""
        pass


class IDecompiler(ABC):
    """Contract for the Decompiler Role (Analyst)."""

    @abstractmethod
    async def decompile(
        self, original_prompt: OriginalPrompt, target_model: Model
    ) -> IntermediateRepresentation:
        """Reverse-engineers the prompt into an IR."""
        pass


class IArchitect(ABC):
    """Contract for the Architect Role (Constructor)."""

    @abstractmethod
    async def design_prompt(
        self, ir: IntermediateRepresentation, target_model: Model, feedback: str | None = None
    ) -> CandidatePrompt:
        """Generates a new prompt based on the IR."""
        pass


class IPilot(ABC):
    """Contract for the Pilot Role (Tester)."""

    @abstractmethod
    async def test_candidate(self, candidate: CandidatePrompt) -> CandidatePrompt:
        """Runs the candidate prompt on the target model."""
        pass


class IJudge(ABC):
    """Contract for the Judge Role (Scorer)."""

    @abstractmethod
    async def evaluate(self, candidate: CandidatePrompt, original: OriginalPrompt) -> float:
        """Compares candidate against baseline and returns a score (0.0 - 1.0)."""
        pass
