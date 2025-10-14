#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Logging Infrastructure with Accelerate Integration

This module provides logging support integrated with HuggingFace Accelerate:
- Accelerate-managed TensorBoard logging
- Accelerate-managed Weights & Biases logging
- Console logging with rich formatting
- Metric tracking and visualization

Key Functions:
    init_trackers: Initialize logging trackers via Accelerate
    log_metrics: Log metrics through Accelerate
    log_hyperparameters: Log training hyperparameters
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .config import LoggingConfig, TrainingConfig


def init_trackers(
    accelerator: Any,
    config: LoggingConfig,
    training_config: TrainingConfig,
    project_name: str = "training",
) -> None:
    """
    Initialize logging trackers via Accelerate.

    Args:
        accelerator: Accelerator instance
        config: Logging configuration
        training_config: Training configuration for hyperparameters
        project_name: Project name for tracking
    """
    # Prepare init kwargs for trackers
    init_kwargs = {}

    # WandB configuration
    if config.wandb:
        wandb_config = {
            "project": config.wandb_project or project_name,
        }
        if config.wandb_entity:
            wandb_config["entity"] = config.wandb_entity
        init_kwargs["wandb"] = wandb_config

    # TensorBoard configuration
    if config.tensorboard:
        init_kwargs["tensorboard"] = {}

    # Initialize trackers through Accelerate
    if init_kwargs:
        accelerator.init_trackers(
            project_name=config.wandb_project or project_name,
            config={
                "epochs": training_config.epochs,
                "batch_size": training_config.batch_size,
                "optimizer": {
                    "type": training_config.optimizer.type,
                    "lr": training_config.optimizer.lr,
                    "weight_decay": training_config.optimizer.weight_decay,
                },
                "scheduler": {
                    "type": training_config.scheduler.type,
                    "warmup_steps": training_config.scheduler.warmup_steps,
                },
                "mixed_precision": training_config.mixed_precision,
                "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "gradient_clipping": training_config.gradient_clipping,
            },
            init_kwargs=init_kwargs,
        )

        accelerator.print(f"Logging initialized: {', '.join(init_kwargs.keys())}")


def log_metrics(
    accelerator: Any,
    metrics: Dict[str, float],
    step: int,
    log_metrics_filter: Optional[list] = None,
) -> None:
    """
    Log metrics through Accelerate.

    Args:
        accelerator: Accelerator instance
        metrics: Dictionary of metrics to log
        step: Current training step
        log_metrics_filter: List of metric name patterns to filter
    """
    # Filter metrics if specified
    if log_metrics_filter:
        metrics = {
            k: v for k, v in metrics.items() if any(m in k for m in log_metrics_filter)
        }

    # Log through Accelerate
    accelerator.log(metrics, step=step)


def log_hyperparameters(
    accelerator: Any,
    hparams: Dict[str, Any],
    save_dir: Optional[str] = None,
) -> None:
    """
    Log hyperparameters.

    Args:
        accelerator: Accelerator instance
        hparams: Dictionary of hyperparameters
        save_dir: Directory to save hyperparameters JSON
    """
    # Save to JSON file
    if save_dir and accelerator.is_main_process:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        hparams_file = save_path / "hyperparameters.json"
        with open(hparams_file, "w") as f:
            json.dump(hparams, f, indent=2)
        accelerator.print(f"Hyperparameters saved: {hparams_file}")


def log_model_summary(
    accelerator: Any,
    model: nn.Module,
    save_dir: Optional[str] = None,
) -> None:
    """
    Log model architecture summary.

    Args:
        accelerator: Accelerator instance
        model: Model to log
        save_dir: Directory to save model summary
    """
    if save_dir and accelerator.is_main_process:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        model_summary_file = save_path / "model_summary.txt"
        with open(model_summary_file, "w") as f:
            f.write(str(model))
            f.write(
                f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}"
            )
            f.write(
                f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
            )

        accelerator.print(f"Model summary saved: {model_summary_file}")


def log_system_info(accelerator: Any):
    """
    Log system and hardware information.

    Args:
        accelerator: Accelerator instance
    """
    import platform

    accelerator.print("\n" + "=" * 70)
    accelerator.print("System Information")
    accelerator.print("=" * 70)
    accelerator.print(f"Platform: {platform.platform()}")
    accelerator.print(f"Python: {platform.python_version()}")
    accelerator.print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        accelerator.print(f"CUDA: {torch.version.cuda}")
        accelerator.print(f"cuDNN: {torch.backends.cudnn.version()}")
        accelerator.print(f"GPU: {torch.cuda.get_device_name(0)}")
        accelerator.print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    accelerator.print("=" * 70 + "\n")
