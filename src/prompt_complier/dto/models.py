"""
Data Transfer Objects (DTOs) and Models for the Prompt Compiler.

This module defines the core data structures used to represent model providers,
prompt styles, and model configurations. It uses `attrs` for data classes and
`marshmallow` for serialization/deserialization.
"""
from enum import Enum
from typing import Any, TypedDict

import marshmallow as ma
from attrs import define, field, validators


class Examples(TypedDict):
    """
    TypedDict representing an input-output example pair.

    Attributes:
        input (str): The input text for the example.
        output (str): The expected output text for the example.
    """
    input: str
    output: str

class ExampleSchema(ma.Schema):
    """
    Marshmallow schema for serializing and deserializing `Examples` TypedDict.
    """
    input = ma.fields.Str(required=True)
    output = ma.fields.Str(required=True)

class ModelProviderType(str, Enum):
    """
    Enumeration of supported model provider types.

    Attributes:
        API: Represents API-based providers like OpenAI, Gemini, Anthropic.
        HUGGINGFACE: Represents Hugging Face models.
    """
    API = "api"             # OpenAI / Gemini / Anthropic
    HUGGINGFACE = "huggingface"

class PromptStyle(str, Enum):
    """
    Enumeration of supported prompt styles.

    Attributes:
        MARKDOWN: Markdown style prompting (preferred by OpenAI / Gemini).
        XML: XML tag based prompting (preferred by Anthropic).
        PLAIN: Plain text prompting (often used for older HF models).
    """
    MARKDOWN = "markdown"   # OpenAI / Gemini prefer this
    XML = "xml"             # Anthropic prefers <instructions> tags
    PLAIN = "plain"         # Older HF models

@define(kw_only=True)
class Provider:
    """
    Represents a model provider configuration.

    Attributes:
        provider (str): The name of the provider (e.g., "OpenAI", "Google").
        provider_type (ModelProviderType): The type of the provider (API or HuggingFace).
        metadata (dict[str, Any]): Additional metadata associated with the provider.
    """
    provider: str = field(validator=validators.instance_of(str))
    provider_type: ModelProviderType = field(validator=validators.instance_of(ModelProviderType))
    metadata: dict[str, Any] = field(factory=dict, validator=validators.instance_of(dict))

class ProviderSchema(ma.Schema):
    """
    Marshmallow schema for serializing and deserializing `Provider` objects.
    """
    provider = ma.fields.Str(required=True)
    provider_type = ma.fields.Enum(ModelProviderType, by_value=True, required=True)
    metadata = ma.fields.Dict(keys=ma.fields.Str(), values=ma.fields.Raw())

    @ma.post_load
    def make_provider(self, data: dict[str, Any], **kwargs: Any) -> Provider:
        """
        Constructs a `Provider` object from deserialized data.

        Args:
            data (dict[str, Any]): The deserialized data.
            **kwargs: Additional keyword arguments.

        Returns:
            Provider: The constructed Provider object.
        """
        return Provider(**data)

@define(kw_only=True)
class Model:
    """
    Represents a specific AI model configuration.

    Attributes:
        provider (Provider): The provider of the model.
        model_name (str): The name of the model (e.g., "gpt-4", "gemini-pro").
        supports_system_messages (bool): Whether the model supports system messages.
        context_window_size (int): The size of the context window in tokens.
        prompt_style (PromptStyle): The preferred prompting style for the model.
        supports_json_mode (bool): Whether the model supports JSON mode output.
        prompting_tips (str): Specific tips or instructions for prompting this model.
    """
    provider: Provider = field(validator=validators.instance_of(Provider))
    model_name: str = field(validator=validators.instance_of(str))
    supports_system_messages: bool = field(validator=validators.instance_of(bool))
    context_window_size: int = field(validator=validators.instance_of(int))
    prompt_style: PromptStyle = field(validator=validators.instance_of(PromptStyle))
    supports_json_mode: bool = field(validator=validators.instance_of(bool))
    prompting_tips: str = field(validator=validators.instance_of(str))
    metadata: dict[str, Any] = field(factory=dict, validator=validators.instance_of(dict))

class ModelSchema(ma.Schema):
    """
    Marshmallow schema for serializing and deserializing `Model` objects.
    """
    provider = ma.fields.Nested(ProviderSchema, required=True)
    model_name = ma.fields.Str(required=True)
    supports_system_messages = ma.fields.Bool(required=True)
    context_window_size = ma.fields.Int(required=True)
    prompt_style = ma.fields.Enum(PromptStyle, by_value=True, required=True)
    supports_json_mode = ma.fields.Bool(required=True)
    prompting_tips = ma.fields.Str(required=True)
    metadata = ma.fields.Dict(keys=ma.fields.Str(), values=ma.fields.Raw())

    @ma.post_load
    def make_model(self, data: dict[str, Any], **kwargs: Any) -> Model:
        """
        Constructs a `Model` object from deserialized data.

        Args:
            data (dict[str, Any]): The deserialized data.
            **kwargs: Additional keyword arguments.

        Returns:
            Model: The constructed Model object.
        """
        return Model(**data)

@define(kw_only=True)
class IntermediateRepresentationMeta:
    """
    Metadata for the Intermediate Representation (IR).

    Attributes:
        source_model (Model): The model that the prompt was originally designed for.
        target_model (Model): The model that the prompt is being compiled for.
    """
    source_model: Model = field(validator=validators.instance_of(Model))
    target_model: Model = field(validator=validators.instance_of(Model))

class IntermediateRepresentationMetaSchema(ma.Schema):
    """
    Marshmallow schema for serializing and deserializing `IntermediateRepresentationMeta` objects.
    """
    source_model = ma.fields.Nested(ModelSchema, required=True)
    target_model = ma.fields.Nested(ModelSchema, required=True)

    @ma.post_load
    def make_ir_meta(self, data: dict[str, Any], **kwargs: Any) -> IntermediateRepresentationMeta:
        """
        Constructs an `IntermediateRepresentationMeta` object from deserialized data.

        Args:
            data (dict[str, Any]): The deserialized data.
            **kwargs: Additional keyword arguments.

        Returns:
            IntermediateRepresentationMeta: The constructed metadata object.
        """
        return IntermediateRepresentationMeta(**data)

@define(kw_only=True)
class IntermediateRepresentationSpec:
    """
    Specification details for the Intermediate Representation (IR).

    Attributes:
        primary_intent (str): The main goal or purpose of the prompt.
        tone_voice (str): The desired tone and voice of the response.
        domain_context (str): The domain or context in which the prompt operates.
        constraints (list[str]): A list of constraints or limitations for the response.
        input_format (str): The expected format of the input.
        output_schema (str): The expected schema or format of the output.
    """
    primary_intent: str = field(validator=validators.instance_of(str))
    tone_voice: str = field(validator=validators.instance_of(str))
    domain_context: str = field(validator=validators.instance_of(str))
    constraints: list[str] = field(factory=list, validator=validators.instance_of(list))
    input_format: str = field(validator=validators.instance_of(str))
    output_schema: str = field(validator=validators.instance_of(str))

class IntermediateRepresentationSpecSchema(ma.Schema):
    """
    Marshmallow schema for serializing and deserializing `IntermediateRepresentationSpec` objects.
    """
    primary_intent = ma.fields.Str(required=True)
    tone_voice = ma.fields.Str(required=True)
    domain_context = ma.fields.Str(required=True)
    constraints = ma.fields.List(ma.fields.Str(), required=True)
    input_format = ma.fields.Str(required=True)
    output_schema = ma.fields.Str(required=True)

    @ma.post_load
    def make_ir_spec(self, data: dict[str, Any], **kwargs: Any) -> IntermediateRepresentationSpec:
        """
        Constructs an `IntermediateRepresentationSpec` object from deserialized data.

        Args:
            data (dict[str, Any]): The deserialized data.
            **kwargs: Additional keyword arguments.

        Returns:
            IntermediateRepresentationSpec: The constructed specification object.
        """
        return IntermediateRepresentationSpec(**data)
    
@define(kw_only=True)
class IntermediateRepresentationData:
    """
    Data component of the Intermediate Representation (IR).

    Attributes:
        few_shot_examples (list[Examples]): A list of few-shot examples to guide the model.
    """
    few_shot_examples: list[Examples] = field(factory=list, validator=validators.deep_iterable(
        member_validator=validators.instance_of(dict),
        iterable_validator=validators.instance_of(list),
  ))

class IntermediateRepresentationDataSchema(ma.Schema):
    """
    Marshmallow schema for serializing and deserializing `IntermediateRepresentationData` objects.
    """
    few_shot_examples = ma.fields.List(
            ma.fields.Nested(ExampleSchema),
            required=True
        )

    @ma.post_load
    def make_ir_data(self, data: dict[str, Any], **kwargs: Any) -> IntermediateRepresentationData:
        """
        Constructs an `IntermediateRepresentationData` object from deserialized data.

        Args:
            data (dict[str, Any]): The deserialized data.
            **kwargs: Additional keyword arguments.

        Returns:
            IntermediateRepresentationData: The constructed data object.
        """
        return IntermediateRepresentationData(**data)
    

@define(kw_only=True)
class IntermediateRepresentation:
    """
    The main Intermediate Representation (IR) object.

    This object encapsulates all the necessary information to represent a prompt
    in a model-agnostic way, including metadata, specifications, and data.

    Attributes:
        meta (IntermediateRepresentationMeta): Metadata about the source and target models.
        spec (IntermediateRepresentationSpec): Specifications for the prompt's intent, tone, etc.
        data (IntermediateRepresentationData): Data associated with the prompt, such as examples.
    """
    meta: IntermediateRepresentationMeta = field(
        validator=validators.instance_of(IntermediateRepresentationMeta)
    )
    spec: IntermediateRepresentationSpec = field(
        validator=validators.instance_of(IntermediateRepresentationSpec)
    )
    data: IntermediateRepresentationData = field(
        validator=validators.instance_of(IntermediateRepresentationData)
    )

class IntermediateRepresentationSchema(ma.Schema):
    """
    Marshmallow schema for serializing and deserializing `IntermediateRepresentation` objects.
    """
    meta = ma.fields.Nested(IntermediateRepresentationMetaSchema, required=True)
    spec = ma.fields.Nested(IntermediateRepresentationSpecSchema, required=True)
    data = ma.fields.Nested(IntermediateRepresentationDataSchema, required=True)

    @ma.post_load
    def make_ir(self, data: dict[str, Any], **kwargs: Any) -> IntermediateRepresentation:
        """
        Constructs an `IntermediateRepresentation` object from deserialized data.

        Args:
            data (dict[str, Any]): The deserialized data.
            **kwargs: Additional keyword arguments.

        Returns:
            IntermediateRepresentation: The constructed IR object.
        """
        return IntermediateRepresentation(**data)