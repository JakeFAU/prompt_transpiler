from prompt_transpiler.dto.models import TokenUsage
from prompt_transpiler.utils.token_collector import TokenCollector

# Constants for test_token_collector_normalization
NORMALIZED_PROMPT_TOKENS = 416  # 30 + 386 (difference added to prompt)
NORMALIZED_COMPLETION_TOKENS = 58
NORMALIZED_TOTAL_TOKENS = 474

# Constants for test_token_collector_accumulation
ACCUMULATED_PROMPT_TOKENS = 15  # 10 + 5
ACCUMULATED_COMPLETION_TOKENS = 15  # 10 + 5
ACCUMULATED_TOTAL_TOKENS = 30  # 20 + 10


def test_token_collector_normalization():
    """
    Test that TokenCollector normalizes inconsistent token usage data.
    Some providers (like Gemini) might report total_tokens > prompt + completion
    due to hidden system prompts or other overhead.
    """
    collector = TokenCollector()
    collector.reset()

    # Simulate inconsistent call:
    # Prompt: 30, Completion: 58, Total: 474
    # Difference: 474 - 88 = 386
    usage1 = TokenUsage(prompt_tokens=30, completion_tokens=58, total_tokens=474)
    collector.add("model-a", usage1)

    summary = collector.get_summary()
    usage = summary["model-a"]

    # Expect prompt tokens to increase by 386 to match total
    assert usage.prompt_tokens == NORMALIZED_PROMPT_TOKENS
    assert usage.completion_tokens == NORMALIZED_COMPLETION_TOKENS
    assert usage.total_tokens == NORMALIZED_TOTAL_TOKENS


def test_token_collector_accumulation():
    """
    Test that TokenCollector correctly accumulates tokens across multiple calls.
    """
    collector = TokenCollector()
    collector.reset()

    usage1 = TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    usage2 = TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10)

    collector.add("model-b", usage1)
    collector.add("model-b", usage2)

    summary = collector.get_summary()
    usage = summary["model-b"]

    assert usage.prompt_tokens == ACCUMULATED_PROMPT_TOKENS
    assert usage.completion_tokens == ACCUMULATED_COMPLETION_TOKENS
    assert usage.total_tokens == ACCUMULATED_TOTAL_TOKENS
