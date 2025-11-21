#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Training Configuration Parser

This module provides parsing for training configurations from YAML files
with support for CLI overrides. Integrates with existing model and data parsers.

Key Functions:
    parse_training_config: Parse training section from YAML
    training_parser: Command-line parser with override support

Example YAML:
    ```yaml
    training:
      epochs: 50
      batch_size: 32
      optimizer:
        type: adamw
        lr: 2e-4
        weight_decay: 0.01
      scheduler:
        type: cosine
        min_lr: 1e-6
        warmup_steps: 1000
    ```
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml
from dacite import from_dict
from yamlargparse import ArgumentParser

from .config import (
    CheckpointConfig,
    DistributedConfig,
    EarlyStoppingConfig,
    LoggingConfig,
    LossConfig,
    OptimizerConfig,
    PerformanceConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)


class TrainingConfigError(Exception):
    """Custom exception for training configuration errors"""

    pass


def _validate_optimizer_config(opt_dict: Dict[str, Any]) -> OptimizerConfig:
    """
    Validate and create OptimizerConfig from dictionary.

    Args:
        opt_dict: Optimizer configuration dictionary

    Returns:
        OptimizerConfig object

    Raises:
        TrainingConfigError: If configuration is invalid
    """
    if "type" not in opt_dict:
        raise TrainingConfigError("Optimizer configuration must include 'type' field")

    if "lr" not in opt_dict:
        raise TrainingConfigError("Optimizer configuration must include 'lr' field")

    # Convert betas list to tuple if present
    if "betas" in opt_dict and isinstance(opt_dict["betas"], list):
        opt_dict["betas"] = tuple(opt_dict["betas"])

    try:
        return from_dict(data_class=OptimizerConfig, data=opt_dict)
    except Exception as e:
        raise TrainingConfigError(f"Invalid optimizer configuration: {e}")


def _validate_scheduler_config(sched_dict: Dict[str, Any]) -> SchedulerConfig:
    """
    Validate and create SchedulerConfig from dictionary.

    Args:
        sched_dict: Scheduler configuration dictionary

    Returns:
        SchedulerConfig object

    Raises:
        TrainingConfigError: If configuration is invalid
    """
    if "type" not in sched_dict:
        raise TrainingConfigError("Scheduler configuration must include 'type' field")

    try:
        return from_dict(data_class=SchedulerConfig, data=sched_dict)
    except Exception as e:
        raise TrainingConfigError(f"Invalid scheduler configuration: {e}")


def parse_training_config(config_path: Union[str, Path]) -> TrainingConfig:
    """
    Parse training configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        TrainingConfig object with validated settings

    Raises:
        TrainingConfigError: If parsing or validation fails

    Example:
        ```python
        from training.parse_training_args import parse_training_config

        config = parse_training_config('configs/experiment.yml')
        print(f"Epochs: {config.epochs}")
        print(f"Learning rate: {config.optimizer.lr}")
        ```
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise TrainingConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise TrainingConfigError(f"Invalid YAML file: {e}")

    if not isinstance(config_data, dict):
        raise TrainingConfigError("Configuration file must contain a dictionary")

    if "training" not in config_data:
        raise TrainingConfigError(
            "Configuration file must contain 'training' section. "
            "See documentation for format."
        )

    training_dict = config_data["training"]
    if not isinstance(training_dict, dict):
        raise TrainingConfigError("'training' section must be a dictionary")

    # Validate required fields
    required = ["epochs", "batch_size", "optimizer", "scheduler"]
    missing = [r for r in required if r not in training_dict]
    if missing:
        raise TrainingConfigError(f"Missing required fields: {', '.join(missing)}")

    # Parse nested configurations
    optimizer_config = _validate_optimizer_config(training_dict.pop("optimizer"))
    scheduler_config = _validate_scheduler_config(training_dict.pop("scheduler"))

    # Parse optional nested configurations
    early_stopping_config = None
    if "early_stopping" in training_dict:
        try:
            early_stopping_config = from_dict(
                data_class=EarlyStoppingConfig, data=training_dict.pop("early_stopping")
            )
        except Exception as e:
            raise TrainingConfigError(f"Invalid early_stopping configuration: {e}")

    validation_config = None
    if "validation" in training_dict:
        try:
            validation_config = from_dict(
                data_class=ValidationConfig, data=training_dict.pop("validation")
            )
        except Exception as e:
            raise TrainingConfigError(f"Invalid validation configuration: {e}")

    checkpoint_config = None
    if "checkpoint" in training_dict:
        try:
            checkpoint_config = from_dict(
                data_class=CheckpointConfig, data=training_dict.pop("checkpoint")
            )
        except Exception as e:
            raise TrainingConfigError(f"Invalid checkpoint configuration: {e}")

    loss_config = None
    if "loss" in training_dict:
        try:
            loss_config = from_dict(
                data_class=LossConfig, data=training_dict.pop("loss")
            )
        except Exception as e:
            raise TrainingConfigError(f"Invalid loss configuration: {e}")

    distributed_config = None
    if "distributed" in training_dict:
        try:
            distributed_config = from_dict(
                data_class=DistributedConfig, data=training_dict.pop("distributed")
            )
        except Exception as e:
            raise TrainingConfigError(f"Invalid distributed configuration: {e}")

    logging_config = None
    if "logging" in training_dict:
        try:
            logging_config = from_dict(
                data_class=LoggingConfig, data=training_dict.pop("logging")
            )
        except Exception as e:
            raise TrainingConfigError(f"Invalid logging configuration: {e}")

    performance_config = None
    if "performance" in training_dict:
        try:
            performance_config = from_dict(
                data_class=PerformanceConfig, data=training_dict.pop("performance")
            )
        except Exception as e:
            raise TrainingConfigError(f"Invalid performance configuration: {e}")

    # Create final training config
    try:
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
            **training_dict,
        )
        return training_config
    except Exception as e:
        raise TrainingConfigError(f"Invalid training configuration: {e}")


def training_parser():
    """
    Parse command line arguments for training configuration with CLI overrides.

    Supports overriding any training parameter from command line for quick
    experimentation without modifying YAML files.

    Returns:
        Tuple of (args, training_config) where:
        - args: Parsed command line arguments
        - training_config: TrainingConfig with CLI overrides applied

    Example:
        ```bash
        # Use config as-is
        python train.py --config configs/experiment.yml

        # Override specific parameters
        python train.py --config configs/experiment.yml \\
            --epochs 100 \\
            --batch-size 64 \\
            --lr 1e-4 \\
            --save-dir ./my_checkpoints
        ```
    """
    parser = ArgumentParser(
        description="Training Configuration Parser with CLI Overrides"
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help='Path to YAML configuration file (must contain "training:" section)',
    )

    # Basic training overrides
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--max-steps", type=int, help="Override max training steps")
    parser.add_argument("--random-seed", type=int, help="Override random seed")

    # Optimizer overrides
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--weight-decay", type=float, help="Override weight decay")

    # Scheduler overrides
    parser.add_argument("--warmup-steps", type=int, help="Override warmup steps")
    parser.add_argument("--min-lr", type=float, help="Override minimum learning rate")

    # Training behavior overrides
    parser.add_argument(
        "--gradient-clipping", type=float, help="Override gradient clipping value"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Enable mixed precision training"
    )
    parser.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
        help="Disable mixed precision training",
    )

    # Checkpoint overrides
    parser.add_argument(
        "--save-dir", type=str, help="Override checkpoint save directory"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, help="Override checkpoint save interval"
    )
    parser.add_argument("--resume", type=str, help="Override checkpoint resume path")

    # Validation overrides
    parser.add_argument("--val-interval", type=int, help="Override validation interval")
    parser.add_argument(
        "--val-batch-size", type=int, help="Override validation batch size"
    )

    # Logging overrides
    parser.add_argument("--log-interval", type=int, help="Override logging interval")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument("--wandb-project", type=str, help="WandB project name")

    # Performance overrides
    parser.add_argument(
        "--num-workers", type=int, help="Override number of DataLoader workers"
    )
    parser.add_argument(
        "--compile-model", action="store_true", help="Enable torch.compile"
    )

    # Feature-specific overrides
    parser.add_argument(
        "--feature-redraw-interval",
        type=int,
        help="Override Performer feature redraw interval",
    )
    parser.add_argument(
        "--fixed-projection",
        action="store_true",
        help="Use fixed projection for Performer",
    )

    # Profiling
    parser.add_argument(
        "--profiling", action="store_true", help="Enable PyTorch profiler"
    )

    # Safeguard overrides
    parser.add_argument(
        "--max-loss",
        type=float,
        help="Threshold for very large loss values used for logging/diagnostics (does NOT skip batches)",
    )

    # Dataset mapping overrides
    parser.add_argument(
        "--train-combine-datasets",
        action="store_true",
        dest="train_combine_datasets",
        help="Combine multiple training datasets",
    )
    parser.add_argument(
        "--no-train-combine-datasets",
        action="store_false",
        dest="train_combine_datasets",
        help="Do not combine multiple training datasets",
    )
    parser.add_argument(
        "--train-splits",
        type=str,
        nargs="+",
        help="List of training splits to use, e.g., 'dset1:split1:0.5 dset2:split2:1.0'",
    )

    args = parser.parse_args()

    try:
        # Parse base configuration from YAML
        training_config = parse_training_config(args.config)

        # Apply CLI overrides (only if provided)
        if args.epochs is not None:
            training_config.epochs = args.epochs
        if args.batch_size is not None:
            training_config.batch_size = args.batch_size
        if args.max_steps is not None:
            training_config.max_steps = args.max_steps
        if args.random_seed is not None:
            training_config.random_seed = args.random_seed

        # Optimizer overrides
        if args.lr is not None:
            training_config.optimizer.lr = args.lr
        if args.weight_decay is not None:
            training_config.optimizer.weight_decay = args.weight_decay

        # Scheduler overrides
        if args.warmup_steps is not None:
            training_config.scheduler.warmup_steps = args.warmup_steps
        if args.min_lr is not None:
            training_config.scheduler.min_lr = args.min_lr

        # Training behavior overrides
        if args.gradient_clipping is not None:
            training_config.gradient_clipping = args.gradient_clipping
        if args.gradient_accumulation_steps is not None:
            training_config.gradient_accumulation_steps = (
                args.gradient_accumulation_steps
            )
        if args.mixed_precision is not None:
            training_config.mixed_precision = args.mixed_precision

        # Checkpoint overrides
        if training_config.checkpoint:
            if args.save_dir is not None:
                training_config.checkpoint.save_dir = args.save_dir
            if args.checkpoint_interval is not None:
                training_config.checkpoint.interval = args.checkpoint_interval
            if args.resume is not None:
                training_config.checkpoint.resume = args.resume

        # Validation overrides
        if training_config.validation:
            if args.val_interval is not None:
                training_config.validation.interval = args.val_interval
            if args.val_batch_size is not None:
                training_config.validation.batch_size = args.val_batch_size

        # Logging overrides
        if training_config.logging:
            if args.log_interval is not None:
                training_config.logging.interval = args.log_interval
            if args.wandb is not None:
                training_config.logging.wandb = args.wandb
            if args.tensorboard is not None:
                training_config.logging.tensorboard = args.tensorboard
            if args.wandb_project is not None:
                training_config.logging.wandb_project = args.wandb_project

        # Performance overrides
        if training_config.performance:
            if args.num_workers is not None:
                training_config.performance.num_workers = args.num_workers
            if args.compile_model is not None:
                training_config.performance.compile_model = args.compile_model

        # Feature-specific overrides
        if args.feature_redraw_interval is not None:
            training_config.feature_redraw_interval = args.feature_redraw_interval
        if args.fixed_projection is not None:
            training_config.fixed_projection = args.fixed_projection

        # Profiling
        if args.profiling is not None:
            training_config.profiling = args.profiling

        # Safeguard overrides
        if getattr(args, "max_loss", None) is not None:
            if training_config.safeguards is None:
                training_config.safeguards = {}
            training_config.safeguards["max_loss"] = args.max_loss

        return args, training_config

    except TrainingConfigError as e:
        parser.error(str(e))


# Example usage
if __name__ == "__main__":
    args, config = training_parser()

    print("=" * 70)
    print("Training Configuration Parsed Successfully")
    print("=" * 70)
    print("\n📈 Training Settings:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Random seed: {config.random_seed}")
    print(f"  Mixed precision: {config.mixed_precision}")

    print("\n🔧 Optimizer:")
    print(f"  Type: {config.optimizer.type}")
    print(f"  Learning rate: {config.optimizer.lr}")
    print(f"  Weight decay: {config.optimizer.weight_decay}")

    print("\n📊 Scheduler:")
    print(f"  Type: {config.scheduler.type}")
    print(f"  Warmup steps: {config.scheduler.warmup_steps}")
    print(f"  Min LR: {config.scheduler.min_lr}")

    if config.checkpoint:
        print("\n💾 Checkpoints:")
        print(f"  Save dir: {config.checkpoint.save_dir}")
        print(f"  Interval: {config.checkpoint.interval}")
        if config.checkpoint.resume:
            print(f"  Resume from: {config.checkpoint.resume}")

    if config.validation:
        print("\n✓ Validation:")
        print(f"  Interval: {config.validation.interval}")
        print(f"  Batch size: {config.validation.batch_size}")

    if config.logging:
        print("\n📝 Logging:")
        print(f"  Interval: {config.logging.interval}")
        print(f"  TensorBoard: {config.logging.tensorboard}")
        print(f"  WandB: {config.logging.wandb}")

    print("\n" + "=" * 70)
