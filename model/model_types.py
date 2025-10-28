#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Model Configuration Types

This module defines configuration dataclasses for model creation and instantiation.
Supports various transformer architectures with typed parameters.

Key Classes:
    ModelConfig: Main configuration container for model creation
    EncoderConfig: Configuration for encoder-only models
    DecoderConfig: Configuration for decoder-only models
    EncoderDecoderConfig: Configuration for seq2seq models
"""

from dataclasses import dataclass
from typing import Literal, Union

from .types import PerformerParams

# Model architecture types
ModelType = Literal["encoder", "decoder", "encoder_decoder"]


@dataclass
class EncoderConfig:
    """
    Configuration for encoder-only models.

    Used for: text classification, embedding extraction, feature encoding

    Inherits all PerformerParams fields for encoder configuration.
    """

    params: PerformerParams

    def to_dict(self) -> dict:
        """Convert to dictionary for model instantiation."""
        return vars(self.params)


@dataclass
class DecoderConfig:
    """
    Configuration for decoder-only models.

    Used for: language modeling, autoregressive generation

    Attributes:
        params: PerformerParams for decoder layers
        use_cross_attention: Whether decoder should support cross-attention
            (True for encoder-decoder, False for decoder-only LM)
    """

    params: PerformerParams
    use_cross_attention: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for model instantiation."""
        config = vars(self.params)
        config["use_cross_attention"] = self.use_cross_attention
        return config


@dataclass
class EncoderDecoderConfig:
    """
    Configuration for encoder-decoder models.

    Used for: translation, summarization, seq2seq tasks

    Attributes:
        encoder_params: PerformerParams for encoder
        decoder_params: PerformerParams for decoder
    """

    encoder_params: PerformerParams
    decoder_params: PerformerParams

    def to_encoder_dict(self) -> dict:
        """Convert encoder params to dictionary."""
        return vars(self.encoder_params)

    def to_decoder_dict(self) -> dict:
        """Convert decoder params to dictionary."""
        config = vars(self.decoder_params)
        # Decoder in encoder-decoder always has cross-attention
        config["use_cross_attention"] = True
        return config


@dataclass
class ModelConfig:
    """
    Main model configuration container.

    This is the top-level configuration object that specifies which type of
    model to create and its parameters.

    Attributes:
        model_type: Type of model ("encoder", "decoder", "encoder_decoder")
        config: Specific configuration (EncoderConfig, DecoderConfig, or EncoderDecoderConfig)
        name: Optional name for the model instance

    Examples:
        ```python
        # Encoder-only model
        model_config = ModelConfig(
            model_type="encoder",
            config=EncoderConfig(params=encoder_params),
            name="bert_classifier"
        )

        # Decoder-only model
        model_config = ModelConfig(
            model_type="decoder",
            config=DecoderConfig(params=decoder_params, use_cross_attention=False),
            name="gpt_lm"
        )

        # Encoder-decoder model
        model_config = ModelConfig(
            model_type="encoder_decoder",
            config=EncoderDecoderConfig(
                encoder_params=enc_params,
                decoder_params=dec_params
            ),
            name="translator"
        )
        ```
    """

    model_type: ModelType
    config: Union[EncoderConfig, DecoderConfig, EncoderDecoderConfig]
    name: str = "transformer"

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.model_type == "encoder" and not isinstance(self.config, EncoderConfig):
            raise ValueError(
                f"model_type 'encoder' requires EncoderConfig, got {type(self.config)}"
            )
        elif self.model_type == "decoder" and not isinstance(
            self.config, DecoderConfig
        ):
            raise ValueError(
                f"model_type 'decoder' requires DecoderConfig, got {type(self.config)}"
            )
        elif self.model_type == "encoder_decoder" and not isinstance(
            self.config, EncoderDecoderConfig
        ):
            raise ValueError(
                f"model_type 'encoder_decoder' requires EncoderDecoderConfig, got {type(self.config)}"
            )
