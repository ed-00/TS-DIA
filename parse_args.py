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

from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

from dataclasses_json import global_config
import yaml
from yamlargparse import ArgumentParser

from data_manager.dataset_types import DatasetConfig, GlobalConfig
from model.parse_model_args import ModelConfigError, parse_model_config
from data_manager.parse_args import DatasetConfigError, parse_dataset_configs
from model.model_types import EncoderConfig, DecoderConfig, EncoderDecoderConfig, ModelConfig

from dacite import from_dict

from training.config import (
    CheckpointConfig,
    DistributedConfig,
    EarlyStoppingConfig,
    LoggingConfig,
    LossConfig,
    PerformanceConfig,
    TrainingConfig,
    TrainingDatasetMap,
    TrainingDatasetSplit,
    ValidationConfig,
)
from training.parse_training_args import (
    TrainingConfigError,
    _validate_optimizer_config,
    _validate_scheduler_config,
)


class ConfigError(Exception):
    """Custom exception for configuration errors"""

    pass


def parse_config(
    config_path: Union[str, Path],
) -> Tuple[Optional[ModelConfig], Optional[List[DatasetConfig]], Optional[GlobalConfig]]:
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
            if model_config is None:
                raise ModelConfigError(
                    "Model configuration parsing returned None")
        except ModelConfigError as e:
            raise ModelConfigError(f"Model configuration error: {e}")

    # Parse dataset configurations if present
    dataset_configs = None
    global_config = None
    if "datasets" in config_data:
        try:
            dataset_configs, global_config = parse_dataset_configs(config_path)
            if dataset_configs is None or len(dataset_configs) == 0:
                raise DatasetConfigError(
                    "Dataset configuration parsing returned empty result"
                )

            # Validate that all configs have global_config attached
            for i, config in enumerate(dataset_configs):
                if not hasattr(config, "global_config") or getattr(config, "global_config", None) is None:
                    raise DatasetConfigError(
                        f"Dataset config {i + 1} ({config.name}) missing global_config. "
                        "Ensure global_config section is present in YAML."
                    )
        except DatasetConfigError as e:
            raise DatasetConfigError(f"Dataset configuration error: {e}")

    # Require at least one section to be present
    if model_config is None and dataset_configs is None:
        raise ConfigError(
            "Configuration file must contain at least one of: 'model' or 'datasets' sections. "
            "Please add the required configuration."
        )

    return model_config, dataset_configs, global_config


def unified_parser() -> Tuple[SimpleNamespace, ModelConfig | None, List[DatasetConfig] | None, GlobalConfig | None, TrainingConfig | None, Path]:
    """
    Unified argument parser for model, dataset, and training configuration.

    All configuration comes from YAML file only, no CLI overrides.

    Returns:
        Tuple of (args, model_config, dataset_configs, training_config, config_path) where:
        - args: Parsed command line arguments
        - model_config: ModelConfig object (None if not in YAML)
        - dataset_configs: List of DatasetConfig objects (None if not in YAML)
        - training_config: TrainingConfig object (None if not in YAML)
        - config_path: Path to the configuration file

    Example:
        ```bash
        # Parse full configuration
        python parse_args.py --config configs/experiment.yml

        # Parse specific sections only
        python parse_args.py --config configs/experiment.yml --model-only
        ```
    """

    parser = ArgumentParser(
        description="Unified Model, Dataset, and Training Configuration Parser"
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help='Path to YAML configuration file (can contain "model:", "datasets:", and/or "training:" sections)',
    )

    # Section filters
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Only parse model configuration (ignore datasets and training)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only parse dataset configuration (ignore model and training)",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Only parse training configuration (ignore model and datasets)",
    )

    # Optional runtime overrides useful for debugging / small experiments
    parser.add_argument(
        "--max-download-files",
        type=int,
        default=None,
        help="Limit number of files to download for each dataset (tries common param names: max_files/limit/num_files)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=None,
        help="Optional override for sampling rate passed to dataset process functions (only applied if supported)",
    )

    args = parser.parse_args()

    try:
        # Parse base configuration from YAML
        model_config, dataset_configs, global_config = parse_config(
            args.config)

        # Parse training config if present
        training_config = None
        config_path = Path(args.config)
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        if "training" in config_data:
            training_dict = config_data["training"]

            # Validate required training fields
            if "optimizer" not in training_dict:
                raise TrainingConfigError(
                    "Training configuration missing required 'optimizer' section"
                )
            if "scheduler" not in training_dict:
                raise TrainingConfigError(
                    "Training configuration missing required 'scheduler' section"
                )

            # Parse nested configurations
            optimizer_config = _validate_optimizer_config(
                training_dict.pop("optimizer")
            )

            scheduler_config = _validate_scheduler_config(
                training_dict.pop("scheduler")
            )

            # Parse optional nested configurations
            early_stopping_config = None
            if "early_stopping" in training_dict:
                early_stopping_config = from_dict(
                    data_class=EarlyStoppingConfig,
                    data=training_dict.pop("early_stopping"),
                )

            validation_config = None
            if "validation" in training_dict:
                validation_dict = training_dict.pop("validation")
                # Parse validation_dataset_map if present
                if "validation_dataset_map" in validation_dict:
                    validation_map_dict = validation_dict.pop(
                        "validation_dataset_map")
                    validation_dataset_map = from_dict(
                        data_class=TrainingDatasetMap,
                        data=validation_map_dict,
                    )
                    validation_dict["validation_dataset_map"] = validation_dataset_map

                validation_config = from_dict(
                    data_class=ValidationConfig, data=validation_dict
                )

            checkpoint_config = None
            if "checkpoint" in training_dict:
                checkpoint_config = from_dict(
                    data_class=CheckpointConfig, data=training_dict.pop(
                        "checkpoint")
                )

            loss_config = None
            if "loss" in training_dict:
                loss_config = from_dict(
                    data_class=LossConfig, data=training_dict.pop("loss")
                )

            distributed_config = None
            if "distributed" in training_dict:
                distributed_config = from_dict(
                    data_class=DistributedConfig, data=training_dict.pop(
                        "distributed")
                )

            logging_config = None
            if "logging" in training_dict:
                logging_config = from_dict(
                    data_class=LoggingConfig, data=training_dict.pop("logging")
                )

            performance_config = None
            if "performance" in training_dict:
                performance_config = from_dict(
                    data_class=PerformanceConfig, data=training_dict.pop(
                        "performance")
                )

            # Parse training_dataset_map if present
            training_dataset_map = None
            if "training_dataset_map" in training_dict:
                training_dataset_map_dict = training_dict.pop(
                    "training_dataset_map")
                training_dataset_map = from_dict(
                    data_class=TrainingDatasetMap,
                    data=training_dataset_map_dict,
                )

            # Create training config
            training_config = TrainingConfig(
                optimizer=optimizer_config,
                scheduler=scheduler_config,
                early_stopping=early_stopping_config,
                validation=validation_config,
                checkpoint=checkpoint_config,
                loss=loss_config,
                distributed=distributed_config,
                logging=logging_config,
                performance=performance_config,
                training_dataset_map=training_dataset_map,
                **training_dict,
            )

        # Apply filters based on flags
        if args.model_only:
            dataset_configs = None
            training_config = None
        if args.data_only:
            model_config = None
            training_config = None
        if args.training_only:
            model_config = None
            dataset_configs = None

        return args, model_config, dataset_configs, global_config, training_config, config_path
    except (
        ConfigError,
        ModelConfigError,
        DatasetConfigError,
        TrainingConfigError,
    ) as e:
        parser.error(str(e))


def combined_parser():
    """
    Legacy parser for backward compatibility.

    Use unified_parser() for new code.
    """
    args, model_config, dataset_configs, global_config, _, _ = unified_parser()
    return args, model_config, dataset_configs


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

        if model_config.model_type == "encoder" and isinstance(model_config.config, EncoderConfig):
            print(
                f"  Encoder layers: {model_config.config.params['num_layers']}")
            print(f"  Model dim: {model_config.config.params['d_model']}")
        elif model_config.model_type == "decoder" and isinstance(model_config.config, DecoderConfig):
            print(
                f"  Decoder layers: {model_config.config.params['num_layers']}")
            print(f"  Model dim: {model_config.config.params['d_model']}")
            print(
                f"  Cross-attention: {model_config.config.use_cross_attention}")
        elif model_config.model_type == "encoder_decoder" and isinstance(model_config.config, EncoderDecoderConfig):
            print(
                f"  Encoder layers: {model_config.config.encoder_params['num_layers']}")
            print(
                f"  Decoder layers: {model_config.config.decoder_params['num_layers']}")
            print(
                f"  Model dim: {model_config.config.encoder_params['d_model']}")

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
