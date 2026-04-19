"""Token usage aggregation utilities."""

from collections import defaultdict
from threading import Lock

from prompt_transpiler.dto.models import TokenUsage


class TokenCollector:
    """
    Singleton class to collect token usage across the application.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls) -> "TokenCollector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.reset()
        return cls._instance

    def reset(self) -> None:
        """Resets the usage statistics."""
        self._usage: dict[str, TokenUsage] = defaultdict(TokenUsage)

    def add(self, model_name: str, usage: TokenUsage) -> None:
        """Adds token usage for a specific model."""
        current = self._usage[model_name]

        p = usage.prompt_tokens
        c = usage.completion_tokens
        t = usage.total_tokens

        # Normalize inconsistent usage data
        # Some providers (like Gemini) might include system prompts in total_tokens
        # but exclude them from prompt_tokens. We attribute the diff to prompt_tokens.
        if t > (p + c):
            p = t - c
        elif t < (p + c):
            t = p + c

        current.prompt_tokens += p
        current.completion_tokens += c
        current.total_tokens += t

    def get_summary(self) -> dict[str, TokenUsage]:
        """Returns a copy of the usage summary."""
        return dict(self._usage)


# Global instance
token_collector = TokenCollector()
