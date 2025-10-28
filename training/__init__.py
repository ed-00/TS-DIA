#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Universal Training Pipeline for Deep Learning Models

This module provides a comprehensive training infrastructure with support for:
- Distributed training (DDP, FSDP)
- Mixed precision training (AMP)
- Advanced optimization and scheduling
- Checkpoint management and resuming
- Custom callbacks and hooks
- Logging and monitoring (TensorBoard, WandB)
- Early stopping and validation
- Feature redrawing for Performer models
- Hyperparameter tuning integration

Main Components:
    - TrainingConfig: Complete training configuration
    - Trainer: Universal training loop
    - Callbacks: Extensible callback system
    - Logging: Multi-backend logging support
"""

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
from .parse_training_args import parse_training_config, training_parser
from .trainer import Trainer

__all__ = [
    # Config classes
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "CheckpointConfig",
    "ValidationConfig",
    "EarlyStoppingConfig",
    "LoggingConfig",
    "PerformanceConfig",
    "DistributedConfig",
    # Parsers
    "parse_training_config",
    "training_parser",
    # Trainer
    "Trainer",
]
