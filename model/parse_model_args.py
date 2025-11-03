#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Model Configuration Parsing

This module provides functions to parse YAML configuration files for model creation.
Similar to data manager parsing, it supports global defaults and type-safe configurations.

Key Functions:
    parse_model_config: Parse YAML file and return ModelConfig object
    validate_model_config: Validate and convert dict to ModelConfig
    model_parser: Command-line argument parser for model configuration

Example YAML Configuration:
    ```yaml
    model:
      model_type: encoder_decoder
      name: translator

      encoder:
        d_model: 512
        num_layers: 6
        num_heads: 8
        d_ff: 4
        dropout: 0.1
        attention_type: linear
        activation: GEGLU
        nb_features: 256

      decoder:
        d_model: 512
        num_layers: 6
        num_heads: 8
        d_ff: 4
        dropout: 0.1
        attention_type: causal_linear
        activation: SWIGLU
        nb_features: 256
        use_cross_attention: true
    ```

Usage:
    ```python
    from model.parse_model_args import parse_model_config
    from model.model_factory import create_model

    # Parse configuration
    config = parse_model_config('configs/model.yml')

    # Create model
    model = create_model(config)
    ```
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml
from yamlargparse import ArgumentParser

from .activations import ActivationFunctions
from .model_types import (
    DecoderConfig,
    EncoderConfig,
    EncoderDecoderConfig,
    ModelConfig,
)
from .types import PerformerParams


class ModelConfigError(Exception):
    """Custom exception for model configuration errors"""

    pass


def _parse_activation(activation_str: str) -> ActivationFunctions:
    """
    Parse activation function string to ActivationFunctions enum.

    Args:
        activation_str: Activation function name (case-insensitive)

    Returns:
        ActivationFunctions enum value

    Raises:
        ModelConfigError: If activation function is unknown
    """
    activation_str = activation_str.upper()
    try:
        return ActivationFunctions[activation_str]
    except KeyError:
        available = ", ".join([a.name for a in ActivationFunctions])
        raise ModelConfigError(
            f"Unknown activation function '{activation_str}'. Available: {available}"
        )


def _parse_device(device_str: Optional[str] = None) -> torch.device:
    """
    Parse device string to torch.device.

    Args:
        device_str: Device string ("cpu", "cuda", "cuda:0", etc.)

    Returns:
        torch.device object
    """
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _create_performer_params(
    params_dict: Dict[str, Any], global_config: Optional[Dict[str, Any]] = None
) -> PerformerParams:
    """
    Create PerformerParams from dictionary with validation.

    Args:
        params_dict: Dictionary of parameters
        global_config: Global configuration defaults

    Returns:
        PerformerParams object

    Raises:
        ModelConfigError: If required parameters are missing or invalid
    """
    # Apply global defaults
    if global_config:
        for key, value in global_config.items():
            if key not in params_dict:
                params_dict[key] = value

    # Required parameters
    required = ["d_model", "num_layers", "num_heads", "d_ff", "dropout", "batch_size"]
    missing = [r for r in required if r not in params_dict]
    if missing:
        raise ModelConfigError(f"Missing required parameters: {', '.join(missing)}")

    # Parse activation function
    if "activation" in params_dict:
        if isinstance(params_dict["activation"], str):
            params_dict["activation"] = _parse_activation(params_dict["activation"])
    else:
        params_dict["activation"] = ActivationFunctions.GELU  # Default

    # Parse device
    params_dict["device"] = _parse_device(params_dict.get("device"))

    # Validate attention_type
    valid_attention = ["softmax", "linear", "causal_linear"]
    attention_type = params_dict.get("attention_type", "softmax")
    if attention_type not in valid_attention:
        raise ModelConfigError(
            f"Invalid attention_type '{attention_type}'. "
            f"Must be one of: {', '.join(valid_attention)}"
        )

    try:
        return PerformerParams(**params_dict)
    except TypeError as e:
        raise ModelConfigError(f"Invalid parameters: {e}")


def validate_model_config(
    model_dict: Dict[str, Any], global_config: Optional[Dict[str, Any]] = None
) -> ModelConfig:
    """
    Validate and convert model configuration dictionary to ModelConfig.

    Args:
        model_dict: Dictionary containing model configuration
        global_config: Global configuration defaults

    Returns:
        ModelConfig object with validated parameters

    Raises:
        ModelConfigError: If validation fails
    """
    if "model_type" not in model_dict:
        raise ModelConfigError("Model configuration must include 'model_type' field")

    model_type = model_dict["model_type"]
    model_name = model_dict.get("name", "transformer")

    # Validate model type
    valid_types = ["encoder", "decoder", "encoder_decoder"]
    if model_type not in valid_types:
        raise ModelConfigError(
            f"Invalid model_type '{model_type}'. "
            f"Must be one of: {', '.join(valid_types)}"
        )

    # Create appropriate config based on model type
    if model_type == "encoder":
        if "encoder" not in model_dict:
            raise ModelConfigError("Encoder model requires 'encoder' configuration")

        encoder_params = _create_performer_params(model_dict["encoder"], global_config)

        config = EncoderConfig(params=encoder_params)

    elif model_type == "decoder":
        if "decoder" not in model_dict:
            raise ModelConfigError("Decoder model requires 'decoder' configuration")

        decoder_dict = model_dict["decoder"]
        use_cross_attention = decoder_dict.pop("use_cross_attention", False)

        decoder_params = _create_performer_params(decoder_dict, global_config)

        config = DecoderConfig(
            params=decoder_params, use_cross_attention=use_cross_attention
        )

    elif model_type == "encoder_decoder":
        if "encoder" not in model_dict or "decoder" not in model_dict:
            raise ModelConfigError(
                "Encoder-decoder model requires both 'encoder' and 'decoder' configurations"
            )

        encoder_params = _create_performer_params(model_dict["encoder"], global_config)

        decoder_dict = model_dict["decoder"]
        # Remove use_cross_attention from decoder dict if present
        # (encoder-decoder always has cross-attention)
        decoder_dict.pop("use_cross_attention", None)

        decoder_params = _create_performer_params(decoder_dict, global_config)

        config = EncoderDecoderConfig(
            encoder_params=encoder_params, decoder_params=decoder_params
        )
    else:
        # This should never happen due to validation above, but ensures config is always defined
        raise ModelConfigError(f"Unsupported model_type: {model_type}")

    return ModelConfig(model_type=model_type, config=config, name=model_name)


def parse_model_config(config_path: Union[str, Path]) -> ModelConfig:
    """
    Parse YAML configuration file and extract model configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated ModelConfig object

    Raises:
        ModelConfigError: If parsing or validation fails

    Example:
        ```python
        config = parse_model_config('configs/model.yml')
        print(f"Model type: {config.model_type}")
        print(f"Model name: {config.name}")
        ```
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ModelConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ModelConfigError(f"Invalid YAML file: {e}")

    if not isinstance(config_data, dict):
        raise ModelConfigError("Configuration file must contain a dictionary")

    if "model" not in config_data:
        raise ModelConfigError(
            "Configuration file must contain 'model' field. Other fields are ignored."
        )

    model_dict = config_data["model"]
    if not isinstance(model_dict, dict):
        raise ModelConfigError("'model' field must be a dictionary")

    # Extract global configuration (shared across encoder/decoder)
    global_config = model_dict.get("global_config", {})

    try:
        return validate_model_config(model_dict, global_config)
    except ModelConfigError as e:
        raise ModelConfigError(f"Model configuration error: {e}")


def model_parser():
    """
    Parse command line arguments for model configuration.

    Returns:
        Tuple of (args, model_config) where:
        - args: Parsed command line arguments
        - model_config: Validated ModelConfig object

    Example:
        ```bash
        python -m model.parse_model_args --config configs/model.yml
        ```
    """
    parser = ArgumentParser(description="Model Configuration Parser")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help='Path to YAML configuration file (only "model:" field is read)',
    )

    args = parser.parse_args()

    try:
        model_config = parse_model_config(args.config)
        return args, model_config
    except ModelConfigError as e:
        parser.error(str(e))


# Example usage
if __name__ == "__main__":
    args, model_config = model_parser()
    print("Parsed arguments:", args)
    print("\nModel configuration:")
    print(f"  Type: {model_config.model_type}")
    print(f"  Name: {model_config.name}")
    print(f"  Config: {model_config.config}")
