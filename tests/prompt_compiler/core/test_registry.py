from prompt_compiler.core.registry import ModelRegistry
from prompt_compiler.dto.models import ModelProviderType, PromptStyle


def test_registry_default_models():
    registry = ModelRegistry()

    # Test OpenAI
    gpt4 = registry.get_model("gpt-4", "openai")
    assert gpt4.model_name == "gpt-4"
    assert gpt4.provider.provider == "openai"
    assert gpt4.prompt_style == PromptStyle.MARKDOWN

    # Test Gemini
    gemini = registry.get_model("gemini-1.5-pro", "google")
    assert gemini.model_name == "gemini-1.5-pro"
    assert gemini.provider.provider == "google"
    assert gemini.prompt_style == PromptStyle.MARKDOWN


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
