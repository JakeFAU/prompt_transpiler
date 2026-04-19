from prompt_transpiler.core.registry import ModelRegistry
from prompt_transpiler.dto.models import ModelProviderType, PromptStyle


def test_registry_default_models():
    registry = ModelRegistry()

    # Test OpenAI
    gpt4o = registry.get_model("gpt-4o", "openai")
    assert gpt4o.model_name == "gpt-4o"
    assert gpt4o.provider.provider == "openai"
    assert gpt4o.prompt_style == PromptStyle.MARKDOWN
    assert gpt4o.supports_system_instructions is True
    assert gpt4o.supports_structured_outputs is True

    # Test Gemini
    gemini = registry.get_model("gemini-2.0-flash", "gemini")
    assert gemini.model_name == "gemini-2.0-flash"
    assert gemini.provider.provider == "gemini"
    assert gemini.prompt_style == PromptStyle.MARKDOWN
    assert gemini.supports_system_instructions is True
    assert gemini.supports_structured_outputs is True

    # Test Claude (Anthropic)
    claude = registry.get_model("claude-3-5-sonnet", "anthropic")
    assert claude.model_name == "claude-3-5-sonnet"
    assert claude.prompt_style == PromptStyle.XML
    assert claude.supports_system_instructions is True
    assert claude.supports_structured_outputs is False


def test_registry_fallback():
    registry = ModelRegistry()

    # Test unknown model fallback
    unknown = registry.get_model("unknown-model", "some-provider")
    assert unknown.model_name == "unknown-model"
    assert unknown.provider.provider == "some-provider"
    assert unknown.provider.provider_type == ModelProviderType.API

    # Test unknown HuggingFace fallback
    hf = registry.get_model("some-bert", "huggingface")
    assert hf.provider.provider_type == ModelProviderType.HUGGINGFACE


def test_registry_register_from_dict():
    registry = ModelRegistry()

    data = {
        "model_name": "custom-model",
        "provider": {
            "provider": "custom",
            "provider_type": "api",
        },
        "supports_system_messages": False,
        "context_window_size": 1024,
        "prompt_style": "plain",
        "supports_json_mode": False,
        "prompting_tips": "None",
    }

    model = registry.register_model_from_dict(data)

    assert model.model_name == "custom-model"
    assert model.provider.provider == "custom"
    assert model.prompt_style == PromptStyle.PLAIN

    # Verify retrieval
    retrieved = registry.get_model("custom-model")
    assert retrieved == model
