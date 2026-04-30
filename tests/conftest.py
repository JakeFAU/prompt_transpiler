import pytest

from prompt_transpiler.dto.models import (
    Model,
    ModelProviderType,
    PromptStyle,
    Provider,
)
from prompt_transpiler.llm.factory import get_llm_provider


@pytest.fixture(autouse=True)
def clear_llm_provider_cache():
    get_llm_provider.cache_clear()



@pytest.fixture
def provider_data():
    return {
        "provider": "OpenAI",
        "provider_type": ModelProviderType.API.value,
        "metadata": {"api_key_env": "OPENAI_API_KEY"},
    }


@pytest.fixture
def provider_obj(provider_data):
    return Provider(
        provider=provider_data["provider"],
        provider_type=ModelProviderType(provider_data["provider_type"]),
        metadata=provider_data["metadata"],
    )


@pytest.fixture
def model_data(provider_data):
    return {
        "provider": provider_data,
        "model_name": "gpt-4",
        "supports_system_messages": True,
        "supports_system_instructions": True,
        "supports_structured_outputs": False,
        "context_window_size": 8192,
        "prompt_style": PromptStyle.MARKDOWN.value,
        "supports_json_mode": True,
        "prompting_tips": "Be concise.",
        "metadata": {"version": "1.0"},
    }


@pytest.fixture
def model_obj(model_data, provider_obj):
    return Model(
        provider=provider_obj,
        model_name=model_data["model_name"],
        supports_system_messages=model_data["supports_system_messages"],
        supports_system_instructions=model_data["supports_system_instructions"],
        supports_structured_outputs=model_data["supports_structured_outputs"],
        context_window_size=model_data["context_window_size"],
        prompt_style=PromptStyle(model_data["prompt_style"]),
        supports_json_mode=model_data["supports_json_mode"],
        prompting_tips=model_data["prompting_tips"],
        metadata=model_data["metadata"],
    )


@pytest.fixture
def example_data():
    return {"input": "Hello", "output": "Hi there!"}


@pytest.fixture
def ir_meta_data(model_data):
    return {
        "source_model": model_data,
        "target_model": model_data,
    }


@pytest.fixture
def ir_spec_data():
    return {
        "primary_intent": "Chat",
        "tone_voice": "Friendly",
        "domain_context": "General",
        "constraints": ["No profanity"],
        "input_format": "Text",
        "output_schema": "Text",
    }


@pytest.fixture
def ir_data_data(example_data):
    return {
        "few_shot_examples": [example_data],
    }


@pytest.fixture
def ir_full_data(ir_meta_data, ir_spec_data, ir_data_data):
    return {
        "meta": ir_meta_data,
        "spec": ir_spec_data,
        "data": ir_data_data,
    }
