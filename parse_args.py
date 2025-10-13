#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Combined Configuration Parser

This module provides a unified parser for both model and dataset configurations.
It reads a single YAML file that can contain both "model:" and "datasets:" sections,
parsing each independently and returning both configurations.

Key Functions:
    parse_config: Parse YAML with both model and dataset configurations
    combined_parser: Command-line parser for complete configuration

Example YAML Configuration:
    ```yaml
    # Model configuration
    model:
      model_type: encoder_decoder
      name: translator

      global_config:
        dropout: 0.1
        batch_size: 32

      encoder:
        d_model: 512
        num_layers: 6
        num_heads: 8
        d_ff: 4
        attention_type: linear
        activation: GEGLU

      decoder:
        d_model: 512
        num_layers: 6
        num_heads: 8
        d_ff: 4
        attention_type: causal_linear
        activation: SWIGLU

    # Dataset configuration
    global_config:
      corpus_dir: ./data
      output_dir: ./manifests
      force_download: false

    datasets:
      - name: librispeech
        download_params:
          dataset_parts: mini_librispeech
      - name: timit
    ```

Usage:
    ```python
    from parse_args import parse_config
    from model.model_factory import create_model
    from data_manager.data_manager import DatasetManager

    # Parse complete configuration
    model_config, dataset_configs = parse_config('configs/experiment.yml')

    # Create model
    model = create_model(model_config)

    # Load datasets
    cut_sets = DatasetManager.load_datasets(datasets=dataset_configs)
    ```

Command Line:
    ```bash
    python parse_args.py --config configs/experiment.yml
    ```
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml
from yamlargparse import ArgumentParser

from data_manager.dataset_types import DatasetConfig
from data_manager.parse_args import DatasetConfigError, parse_dataset_configs
from model.model_types import ModelConfig
from model.parse_model_args import ModelConfigError, parse_model_config


class ConfigError(Exception):
    """Custom exception for configuration errors"""

    pass


def parse_config(
    config_path: Union[str, Path],
) -> Tuple[Optional[ModelConfig], Optional[List[DatasetConfig]]]:
    """
    Parse YAML configuration file with both model and dataset configurations.

    This function reads a single YAML file and extracts both model and dataset
    configurations. Either section can be omitted if not needed.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (model_config, dataset_configs) where:
        - model_config: ModelConfig object (None if not present)
        - dataset_configs: List of DatasetConfig objects (None if not present)

    Raises:
        ConfigError: If file not found or YAML is invalid
        ModelConfigError: If model configuration is invalid
        DatasetConfigError: If dataset configuration is invalid

    Example:
        ```python
        from parse_args import parse_config

        # Parse configuration (handles missing sections gracefully)
        model_config, dataset_configs = parse_config('configs/full.yml')

        if model_config:
            print(f"Model: {model_config.name}")

        if dataset_configs:
            print(f"Datasets: {[d.name for d in dataset_configs]}")
        ```
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML file: {e}")

    if not isinstance(config_data, dict):
        raise ConfigError("Configuration file must contain a dictionary")

    # Parse model configuration if present
    model_config = None
    if "model" in config_data:
        try:
            model_config = parse_model_config(config_path)
        except ModelConfigError as e:
            raise ModelConfigError(f"Model configuration error: {e}")

    # Parse dataset configurations if present
    dataset_configs = None
    if "datasets" in config_data:
        try:
            dataset_configs = parse_dataset_configs(config_path)
        except DatasetConfigError as e:
            raise DatasetConfigError(f"Dataset configuration error: {e}")

    # Warn if neither section is present
    if model_config is None and dataset_configs is None:
        print(
            "Warning: Configuration file contains neither 'model' nor 'datasets' sections. "
            "At least one is recommended."
        )

    return model_config, dataset_configs


def combined_parser():
    """
    Parse command line arguments for combined model and dataset configuration.

    Returns:
        Tuple of (args, model_config, dataset_configs) where:
        - args: Parsed command line arguments
        - model_config: ModelConfig object (None if not in YAML)
        - dataset_configs: List of DatasetConfig objects (None if not in YAML)

    Example:
        ```bash
        # Parse configuration with both model and datasets
        python parse_args.py --config configs/experiment.yml

        # Also works with only model or only datasets in YAML
        python parse_args.py --config configs/model_only.yml
        python parse_args.py --config configs/datasets_only.yml
        ```
    """
    parser = ArgumentParser(
        description="Combined Model and Dataset Configuration Parser"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help='Path to YAML configuration file (can contain "model:" and/or "datasets:" sections)',
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Only parse model configuration (ignore datasets)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only parse dataset configuration (ignore model)",
    )

    args = parser.parse_args()

    try:
        model_config, dataset_configs = parse_config(args.config)

        # Apply filters based on flags
        if args.model_only:
            dataset_configs = None
        if args.data_only:
            model_config = None

        return args, model_config, dataset_configs
    except (ConfigError, ModelConfigError, DatasetConfigError) as e:
        parser.error(str(e))


def create_example_config(output_path: Union[str, Path] = "configs/example.yml"):
    """
    Create an example configuration file with both model and datasets.

    Args:
        output_path: Path where to save the example configuration

    Example:
        ```python
        from parse_args import create_example_config

        create_example_config("configs/my_experiment.yml")
        ```
    """
    example_config = {
        "model": {
            "model_type": "encoder_decoder",
            "name": "my_translator",
            "global_config": {"dropout": 0.1, "batch_size": 32, "d_ff": 4},
            "encoder": {
                "d_model": 512,
                "num_layers": 6,
                "num_heads": 8,
                "attention_type": "linear",
                "activation": "GEGLU",
                "nb_features": 256,
            },
            "decoder": {
                "d_model": 512,
                "num_layers": 6,
                "num_heads": 8,
                "attention_type": "causal_linear",
                "activation": "SWIGLU",
                "nb_features": 256,
                "use_cross_attention": True,
            },
        },
        "global_config": {
            "corpus_dir": "./data",
            "output_dir": "./manifests",
            "force_download": False,
        },
        "datasets": [
            {"name": "yesno"},
            {"name": "timit", "process_params": {"num_phones": 48}},
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)

    print(f"Example configuration created at: {output_path}")


# Example usage
if __name__ == "__main__":
    args, model_config, dataset_configs = combined_parser()

    print("=" * 70)
    print("Configuration Parsed Successfully")
    print("=" * 70)

    if model_config:
        print("\n📦 Model Configuration:")
        print(f"  Type: {model_config.model_type}")
        print(f"  Name: {model_config.name}")

        if model_config.model_type == "encoder":
            print(f"  Encoder layers: {model_config.config.params.num_layers}")
            print(f"  Model dim: {model_config.config.params.d_model}")
        elif model_config.model_type == "decoder":
            print(f"  Decoder layers: {model_config.config.params.num_layers}")
            print(f"  Model dim: {model_config.config.params.d_model}")
            print(f"  Cross-attention: {model_config.config.use_cross_attention}")
        elif model_config.model_type == "encoder_decoder":
            print(f"  Encoder layers: {model_config.config.encoder_params.num_layers}")
            print(f"  Decoder layers: {model_config.config.decoder_params.num_layers}")
            print(f"  Model dim: {model_config.config.encoder_params.d_model}")

    if dataset_configs:
        print("\n📊 Dataset Configuration:")
        print(f"  Number of datasets: {len(dataset_configs)}")
        for i, config in enumerate(dataset_configs, 1):
            print(f"  {i}. {config.name}")

    if not model_config and not dataset_configs:
        print("\n⚠️  No configurations found in file")

    print("\n" + "=" * 70)

    # Optionally create example config
    if args.config == "create_example":
        create_example_config()
