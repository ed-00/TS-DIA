#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Model Factory Pattern

This module provides a factory pattern for creating transformer models from configurations.
Similar to DatasetManager, it handles model instantiation with type safety and validation.

Key Classes:
    ModelFactory: Main factory class for creating models

Functions:
    create_model: Factory method to instantiate models from configuration

Usage Examples:
    ```python
    from model.model_factory import ModelFactory
    from model.parse_model_args import parse_model_config

    # Load from configuration file
    model_config = parse_model_config('configs/model.yml')
    model = ModelFactory.create_model(model_config)

    # Or create directly
    from model.model_types import ModelConfig, EncoderConfig

    config = ModelConfig(
        model_type="encoder",
        config=EncoderConfig(params=encoder_params)
    )
    model = ModelFactory.create_model(config)
    ```
"""

from typing import Union

from .model_types import (
    DecoderConfig,
    EncoderConfig,
    EncoderDecoderConfig,
    ModelConfig,
)
from .transformer import EncoderDecoderTransformer, PerformerDecoder, PerformerEncoder


class ModelFactory:
    """
    Factory for creating transformer models from configurations.

    This factory provides a unified interface for creating different types of
    transformer models (encoder, decoder, encoder-decoder) from typed configurations.

    Features:
    - Type-safe model creation
    - Automatic parameter validation
    - Support for all transformer variants
    - Extensible architecture

    Examples:
        ```python
        # Create encoder
        encoder = ModelFactory.create_encoder(encoder_config)

        # Create decoder
        decoder = ModelFactory.create_decoder(decoder_config)

        # Create encoder-decoder
        seq2seq = ModelFactory.create_encoder_decoder(enc_dec_config)

        # Create from ModelConfig (auto-detects type)
        model = ModelFactory.create_model(model_config)
        ```
    """

    @staticmethod
    def create_encoder(config: EncoderConfig) -> PerformerEncoder:
        """
        Create an encoder-only model.

        Args:
            config: EncoderConfig with model parameters

        Returns:
            PerformerEncoder: Initialized encoder model

        Example:
            ```python
            from model.types import PerformerParams
            from model.activations import ActivationFunctions

            encoder_params = PerformerParams(
                d_model=512,
                device=device,
                batch_size=32,
                num_layers=6,
                num_heads=8,
                d_ff=4,
                dropout=0.1,
                attention_type="linear",
                activation=ActivationFunctions.GEGLU
            )

            encoder_config = EncoderConfig(params=encoder_params)
            encoder = ModelFactory.create_encoder(encoder_config)
            ```
        """
        params_dict = config.to_dict()
        # Ensure encoder doesn't have cross-attention
        params_dict["use_cross_attention"] = False
        return PerformerEncoder(**params_dict)

    @staticmethod
    def create_decoder(config: DecoderConfig) -> PerformerDecoder:
        """
        Create a decoder model.

        Args:
            config: DecoderConfig with model parameters

        Returns:
            PerformerDecoder: Initialized decoder model

        Example:
            ```python
            decoder_params = PerformerParams(
                d_model=512,
                device=device,
                batch_size=32,
                num_layers=6,
                num_heads=8,
                d_ff=4,
                dropout=0.1,
                attention_type="causal_linear",
                activation=ActivationFunctions.SWIGLU
            )

            # Decoder-only (no cross-attention)
            decoder_config = DecoderConfig(
                params=decoder_params,
                use_cross_attention=False
            )
            decoder = ModelFactory.create_decoder(decoder_config)
            ```
        """
        params_dict = config.to_dict()
        return PerformerDecoder(**params_dict)

    @staticmethod
    def create_encoder_decoder(
        config: EncoderDecoderConfig,
    ) -> EncoderDecoderTransformer:
        """
        Create an encoder-decoder model.

        Args:
            config: EncoderDecoderConfig with encoder and decoder parameters

        Returns:
            EncoderDecoderTransformer: Initialized encoder-decoder model

        Example:
            ```python
            enc_params = PerformerParams(...)
            dec_params = PerformerParams(...)

            enc_dec_config = EncoderDecoderConfig(
                encoder_params=enc_params,
                decoder_params=dec_params
            )

            seq2seq = ModelFactory.create_encoder_decoder(enc_dec_config)
            ```
        """
        encoder_dict = config.to_encoder_dict()
        decoder_dict = config.to_decoder_dict()

        return EncoderDecoderTransformer(
            encoder_params=encoder_dict, decoder_params=decoder_dict
        )

    @staticmethod
    def create_model(
        config: ModelConfig,
    ) -> Union[PerformerEncoder, PerformerDecoder, EncoderDecoderTransformer]:
        """
        Create a model from ModelConfig (auto-detects type).

        This is the main factory method that automatically creates the appropriate
        model type based on the configuration.

        Args:
            config: ModelConfig specifying model type and parameters

        Returns:
            Union[PerformerEncoder, PerformerDecoder, EncoderDecoderTransformer]:
                The instantiated model

        Raises:
            ValueError: If configuration is invalid or model type is unknown

        Example:
            ```python
            from model.parse_model_args import parse_model_config

            # Load from YAML
            model_config = parse_model_config('configs/model.yml')
            model = ModelFactory.create_model(model_config)

            # Model type is automatically determined
            if isinstance(model, PerformerEncoder):
                print("Created encoder")
            elif isinstance(model, PerformerDecoder):
                print("Created decoder")
            elif isinstance(model, EncoderDecoderTransformer):
                print("Created encoder-decoder")
            ```
        """
        # Validate configuration
        config.validate()

        # Create appropriate model type
        if config.model_type == "encoder":
            return ModelFactory.create_encoder(config.config)
        elif config.model_type == "decoder":
            return ModelFactory.create_decoder(config.config)
        elif config.model_type == "encoder_decoder":
            return ModelFactory.create_encoder_decoder(config.config)
        else:
            raise ValueError(
                f"Unknown model type: {config.model_type}. "
                f"Must be one of: 'encoder', 'decoder', 'encoder_decoder'"
            )

    @staticmethod
    def list_model_types() -> list[str]:
        """
        List all available model types.

        Returns:
            List of supported model type strings
        """
        return ["encoder", "decoder", "encoder_decoder"]


def create_model(
    config: ModelConfig,
) -> Union[PerformerEncoder, PerformerDecoder, EncoderDecoderTransformer]:
    """
    Convenience function for creating models.

    This is a wrapper around ModelFactory.create_model for simpler imports.

    Args:
        config: ModelConfig specifying model type and parameters

    Returns:
        Union[PerformerEncoder, PerformerDecoder, EncoderDecoderTransformer]:
            The instantiated model

    Example:
        ```python
        from model.model_factory import create_model
        from model.parse_model_args import parse_model_config

        config = parse_model_config('configs/model.yml')
        model = create_model(config)
        ```
    """
    return ModelFactory.create_model(config)
