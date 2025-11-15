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
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from .config import LoggingConfig, TrainingConfig


def init_trackers(
    accelerator:  Accelerator,
    config: LoggingConfig,
    training_config: TrainingConfig,
    project_name: str = "training",
    config_path: Optional[str] = None,
) -> None:
    """
    Initialize logging trackers via Accelerate.

    Args:
        accelerator: Accelerator instance
        config: Logging configuration
        training_config: Training configuration for hyperparameters
        project_name: Project name for tracking
        config_path: Path to configuration YAML file (for sending full config to wandb)
    """
    # Prepare init kwargs for trackers
    init_kwargs = {}

    # WandB configuration
    if config.wandb:
        wandb_config = {}
        # Note: project is passed via project_name parameter to init_trackers, not here
        if config.wandb_entity:
            wandb_config["entity"] = config.wandb_entity
        init_kwargs["wandb"] = wandb_config

    # TensorBoard configuration
    if config.tensorboard:
        init_kwargs["tensorboard"] = {}

    # Initialize trackers through Accelerate
    if init_kwargs:
        # Load full config from YAML file if available
        full_config = {}
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    full_config = yaml.safe_load(f)
            except Exception as e:
                accelerator.print(
                    f"Warning: Could not load full config from {config_path}: {e}"
                )
                # Fall back to flattened config
                full_config = {
                    "epochs": training_config.epochs,
                    "batch_size": training_config.batch_size,
                    "optimizer_type": training_config.optimizer.type,
                    "optimizer_lr": training_config.optimizer.lr,
                    "optimizer_weight_decay": training_config.optimizer.weight_decay,
                    "scheduler_type": training_config.scheduler.type,
                    "scheduler_warmup_steps": training_config.scheduler.warmup_steps,
                    "mixed_precision": str(training_config.mixed_precision),
                    "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                    "gradient_clipping": training_config.gradient_clipping,
                }
        else:
            # Use flattened config if no config_path provided
            full_config = {
                "epochs": training_config.epochs,
                "batch_size": training_config.batch_size,
                "optimizer_type": training_config.optimizer.type,
                "optimizer_lr": training_config.optimizer.lr,
                "optimizer_weight_decay": training_config.optimizer.weight_decay,
                "scheduler_type": training_config.scheduler.type,
                "scheduler_warmup_steps": training_config.scheduler.warmup_steps,
                "mixed_precision": str(training_config.mixed_precision),
                "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "gradient_clipping": training_config.gradient_clipping,
            }

        accelerator.init_trackers(
            project_name=config.wandb_project or project_name,
            config=full_config,
            init_kwargs=init_kwargs,
        )

        accelerator.print(
            f"Logging initialized: {', '.join(init_kwargs.keys())}")


def log_metrics(
    accelerator:  Accelerator,
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
    accelerator:  Accelerator,
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
    accelerator:  Accelerator,
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


def log_system_info(accelerator: Accelerator):
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
        accelerator.print(f"CUDA: {torch.cuda._parse_visible_devices()}")
        accelerator.print(f"cuDNN: {torch.backends.cudnn.version()}")
        accelerator.print(f"GPU: {torch.cuda.get_device_name(0)}")
        accelerator.print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    accelerator.print("=" * 70 + "\n")


def save_wandb_info(accelerator:  Accelerator, checkpoint_dir: str) -> None:
    """
    Save WandB run information to checkpoint directory.

    Args:
        accelerator: Accelerator instance
        checkpoint_dir: Directory to save wandb info
    """
    if not accelerator.is_main_process:
        return

    # Check if wandb tracker is active
    wandb_tracker = None
    if hasattr(accelerator, "trackers"):
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                wandb_tracker = tracker.tracker
                break

    if wandb_tracker is None:
        return

    try:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        wandb_info = {
            "run_id": wandb_tracker.id,
            "run_name": wandb_tracker.name,
            "run_url": wandb_tracker.url,
            "project": wandb_tracker.project,
            "entity": wandb_tracker.entity,
            "timestamp": datetime.now().isoformat(),
        }

        wandb_info_path = checkpoint_path / "wandb_info.json"
        with open(wandb_info_path, "w") as f:
            json.dump(wandb_info, f, indent=2)

        accelerator.print(f"WandB info saved: {wandb_info_path}")
        accelerator.print(
            f"  Run: {wandb_info['run_name']} ({wandb_info['run_id']})")
        accelerator.print(f"  URL: {wandb_info['run_url']}")
    except Exception as e:
        accelerator.print(f"Warning: Could not save wandb info: {e}")


def setup_file_logger(checkpoint_dir: str, is_resume: bool = False) -> str:
    """
    Setup file logging to capture all console output.

    Args:
        checkpoint_dir: Directory to save log files
        is_resume: Whether this is a resumed training run

    Returns:
        Path to the log file
    """
    logs_dir = Path(checkpoint_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if is_resume:
        # Count existing resume logs
        resume_logs = list(logs_dir.glob("training_resume_*.log"))
        resume_count = len(resume_logs) + 1
        log_filename = f"training_resume_{resume_count}_{timestamp}.log"
    else:
        log_filename = f"training_{timestamp}.log"

    log_path = logs_dir / log_filename

    # Setup Python logging to write to both console and file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.info(f"Logging to file: {log_path}")

    return str(log_path)
