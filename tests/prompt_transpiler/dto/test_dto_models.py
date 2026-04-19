import marshmallow as ma
import pytest

from prompt_transpiler.dto.models import (
    ExampleSchema,
    IntermediateRepresentation,
    IntermediateRepresentationDataSchema,
    IntermediateRepresentationMetaSchema,
    IntermediateRepresentationSchema,
    IntermediateRepresentationSpecSchema,
    Message,
    Model,
    ModelProviderType,
    ModelSchema,
    PromptPayload,
    PromptPayloadSchema,
    PromptStyle,
    Provider,
    ProviderSchema,
)


class TestProvider:
    def test_provider_deserialization(self, provider_data):
        schema = ProviderSchema()
        provider = schema.load(provider_data)
        assert isinstance(provider, Provider)
        assert provider.provider == "OpenAI"
        assert provider.provider_type == ModelProviderType.API
        assert provider.metadata == {"api_key_env": "OPENAI_API_KEY"}

    def test_provider_serialization(self, provider_obj, provider_data):
        schema = ProviderSchema()
        dumped = schema.dump(provider_obj)
        assert dumped == provider_data

    def test_provider_invalid_type(self, provider_data):
        data = provider_data.copy()
        data["provider_type"] = "invalid_type"
        schema = ProviderSchema()
        with pytest.raises(ma.ValidationError) as excinfo:
            schema.load(data)
        assert "provider_type" in excinfo.value.messages

    def test_provider_missing_field(self, provider_data):
        data = provider_data.copy()
        del data["provider"]
        schema = ProviderSchema()
        with pytest.raises(ma.ValidationError) as excinfo:
            schema.load(data)
        assert "provider" in excinfo.value.messages


class TestModel:
    def test_model_deserialization(self, model_data):
        schema = ModelSchema()
        model = schema.load(model_data)
        assert isinstance(model, Model)
        assert model.model_name == "gpt-4"
        assert model.prompt_style == PromptStyle.MARKDOWN
        assert model.supports_system_instructions is True
        assert model.supports_structured_outputs is False
        assert isinstance(model.provider, Provider)

    def test_model_serialization(self, model_obj, model_data):
        schema = ModelSchema()
        dumped = schema.dump(model_obj)
        assert dumped == model_data

    def test_model_missing_nested_field(self, model_data):
        data = model_data.copy()
        # removing 'provider' field entirely
        del data["provider"]
        schema = ModelSchema()
        with pytest.raises(ma.ValidationError) as excinfo:
            schema.load(data)
        assert "provider" in excinfo.value.messages


class TestExamples:
    def test_example_serialization(self, example_data):
        schema = ExampleSchema()
        loaded = schema.load(example_data)
        assert loaded["input"] == "Hello"  # pyright: ignore
        assert loaded["output"] == "Hi there!"  # pyright: ignore

        dumped = schema.dump(loaded)
        assert dumped == example_data


class TestIntermediateRepresentation:
    def test_ir_meta_deserialization(self, ir_meta_data):
        schema = IntermediateRepresentationMetaSchema()
        meta = schema.load(ir_meta_data)
        assert isinstance(meta.source_model, Model)  # pyright: ignore
        assert isinstance(meta.target_model, Model)  # pyright: ignore

    def test_ir_spec_deserialization(self, ir_spec_data):
        schema = IntermediateRepresentationSpecSchema()
        spec = schema.load(ir_spec_data)
        assert spec.primary_intent == "Chat"  # pyright: ignore
        assert spec.constraints == ["No profanity"]  # pyright: ignore

    def test_ir_data_deserialization(self, ir_data_data):
        schema = IntermediateRepresentationDataSchema()
        data = schema.load(ir_data_data)
        assert len(data.few_shot_examples) == 1  # pyright: ignore
        assert data.few_shot_examples[0]["input"] == "Hello"  # pyright: ignore

    def test_ir_full_cycle(self, ir_full_data):
        schema = IntermediateRepresentationSchema()
        # Test Deserialization
        ir_obj = schema.load(ir_full_data)
        assert isinstance(ir_obj, IntermediateRepresentation)
        assert ir_obj.spec.tone_voice == "Friendly"

        # Test Serialization
        dumped = schema.dump(ir_obj)
        assert dumped == ir_full_data

    def test_ir_validation_error(self, ir_full_data):
        data = ir_full_data.copy()
        # Invalidate one part (spec)
        del data["spec"]["primary_intent"]
        schema = IntermediateRepresentationSchema()
        with pytest.raises(ma.ValidationError) as excinfo:
            schema.load(data)
        # Marshmallow nested validation errors are usually keyed by the field name
        assert "spec" in excinfo.value.messages


class TestPromptPayload:
    def test_prompt_payload_serialization(self):
        payload = PromptPayload(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello!"),
            ],
            response_format={"type": "json_object"},
        )
        schema = PromptPayloadSchema()
        data = schema.dump(payload)
        assert data["messages"][0]["role"] == "system"
        assert data["messages"][0]["content"] == "You are a helpful assistant."
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"] == "Hello!"
        assert data["response_format"]["type"] == "json_object"

        loaded = schema.load(data)
        assert isinstance(loaded, PromptPayload)
        expected_count = 2
        assert len(loaded.messages) == expected_count
        assert loaded.messages[0].role == "system"
        assert loaded.messages[0].content == "You are a helpful assistant."
        assert loaded.messages[1].role == "user"
        assert loaded.messages[1].content == "Hello!"
        assert loaded.response_format == {"type": "json_object"}

    def test_prompt_payload_full_text(self):
        payload = PromptPayload(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello!"),
            ]
        )
        expected = "system: You are a helpful assistant.\nuser: Hello!"
        assert payload.full_text == expected
