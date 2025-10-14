#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
HuggingFace Accelerate Integration

This module provides utilities for setting up and using HuggingFace Accelerate
for distributed training, mixed precision, and gradient accumulation.

Key Functions:
    setup_accelerator: Create and configure Accelerator
    prepare_for_training: Prepare model, optimizer, and dataloaders
    save_checkpoint_with_accelerate: Save checkpoint using Accelerate
    load_checkpoint_with_accelerate: Load checkpoint using Accelerate
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from .config import TrainingConfig


def setup_accelerator(
    training_config: TrainingConfig,
    project_dir: Optional[str] = None,
    logging_dir: Optional[str] = None,
) -> Accelerator:
    """
    Setup HuggingFace Accelerator with configuration from TrainingConfig.

    Args:
        training_config: Training configuration
        project_dir: Project directory for outputs
        logging_dir: Directory for logs

    Returns:
        Configured Accelerator instance

    Example:
        ```python
        from training.accelerate_utils import setup_accelerator
        from training.parse_training_args import parse_training_config

        config = parse_training_config('config.yml')
        accelerator = setup_accelerator(config)
        ```
    """
    # Determine mixed precision mode
    mixed_precision = "no"
    if training_config.mixed_precision:
        mixed_precision = "fp16"  # Can be extended to support bf16

    # Setup project configuration for better organization
    project_config = None
    if project_dir or (
        training_config.checkpoint and training_config.checkpoint.save_dir
    ):
        save_dir = project_dir or training_config.checkpoint.save_dir
        project_config = ProjectConfiguration(
            project_dir=save_dir,
            logging_dir=logging_dir or os.path.join(save_dir, "logs"),
            automatic_checkpoint_naming=True,
            total_limit=training_config.checkpoint.save_total_limit
            if training_config.checkpoint
            else None,
        )

    # Create accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        log_with=_get_log_trackers(training_config),
        project_config=project_config,
        cpu=(not torch.cuda.is_available()),
    )

    # Initialize distributed backend if configured
    if training_config.distributed and training_config.distributed.world_size > 1:
        if accelerator.state.distributed_type.value == "NO":
            accelerator.print(
                "Warning: Distributed config specified but Accelerate is not in distributed mode. "
                "Make sure to launch with accelerate launch or torchrun."
            )

    return accelerator


def _get_log_trackers(training_config: TrainingConfig) -> list:
    """Get list of logging trackers to use."""
    trackers = []
    if training_config.logging:
        if training_config.logging.tensorboard:
            trackers.append("tensorboard")
        if training_config.logging.wandb:
            trackers.append("wandb")
    return trackers if trackers else None


def prepare_for_training(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    lr_scheduler: Optional[Any] = None,
) -> Tuple:
    """
    Prepare model, optimizer, and dataloaders for training with Accelerate.

    Args:
        accelerator: Accelerator instance
        model: PyTorch model
        optimizer: Optimizer
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader (optional)
        lr_scheduler: Learning rate scheduler (optional)

    Returns:
        Tuple of prepared (model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    Example:
        ```python
        model, optimizer, train_dl, val_dl, scheduler = prepare_for_training(
            accelerator, model, optimizer, train_dl, val_dl, scheduler
        )
        ```
    """
    # Prepare objects for distributed/mixed precision
    if lr_scheduler is not None:
        if val_dataloader is not None:
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
                accelerator.prepare(
                    model, optimizer, train_dataloader, val_dataloader, lr_scheduler
                )
            )
        else:
            model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, lr_scheduler
            )
            val_dataloader = None
    else:
        if val_dataloader is not None:
            model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, val_dataloader
            )
        else:
            model, optimizer, train_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader
            )
            val_dataloader = None
        lr_scheduler = None

    return model, optimizer, train_dataloader, val_dataloader, lr_scheduler


def save_checkpoint_with_accelerate(
    accelerator: Accelerator,
    save_dir: str,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[Any] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> str:
    """
    Save checkpoint using Accelerate's state management.

    Args:
        accelerator: Accelerator instance
        save_dir: Directory to save checkpoint
        epoch: Current epoch
        step: Current step
        model: Model to save
        optimizer: Optimizer to save
        lr_scheduler: Scheduler to save (optional)
        extra_state: Additional state to save (optional)
        is_best: Whether this is the best checkpoint

    Returns:
        Path to saved checkpoint

    Example:
        ```python
        checkpoint_path = save_checkpoint_with_accelerate(
            accelerator, './checkpoints', epoch=10, step=1000,
            model=model, optimizer=optimizer, lr_scheduler=scheduler
        )
        ```
    """
    save_path = Path(save_dir) / f"checkpoint-epoch{epoch}-step{step}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save using Accelerate's save_state
    accelerator.save_state(str(save_path))

    # Save extra state if provided
    if extra_state and accelerator.is_main_process:
        extra_state_path = save_path / "extra_state.pt"
        torch.save(extra_state, extra_state_path)

    # Save best checkpoint marker
    if is_best and accelerator.is_main_process:
        best_marker = save_path / "BEST_CHECKPOINT"
        best_marker.touch()

        # Also create a symlink to the best checkpoint
        best_link = Path(save_dir) / "best_checkpoint"
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        best_link.symlink_to(save_path.name)

    accelerator.print(f"Checkpoint saved: {save_path}")
    return str(save_path)


def load_checkpoint_with_accelerate(
    accelerator: Accelerator,
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint using Accelerate's state management.

    Args:
        accelerator: Accelerator instance
        checkpoint_path: Path to checkpoint directory
        model: Model to load into
        optimizer: Optimizer to load into
        lr_scheduler: Scheduler to load into (optional)

    Returns:
        Dictionary with extra state (if any)

    Example:
        ```python
        extra_state = load_checkpoint_with_accelerate(
            accelerator, './checkpoints/checkpoint-epoch10-step1000',
            model=model, optimizer=optimizer, lr_scheduler=scheduler
        )
        start_epoch = extra_state.get('epoch', 0)
        ```
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load using Accelerate's load_state
    accelerator.load_state(str(checkpoint_path))

    # Load extra state if exists
    extra_state = {}
    extra_state_path = checkpoint_path / "extra_state.pt"
    if extra_state_path.exists():
        extra_state = torch.load(extra_state_path, map_location=accelerator.device)

    accelerator.print(f"Checkpoint loaded: {checkpoint_path}")
    return extra_state


def get_gradient_norm(
    accelerator: Accelerator,
    model: torch.nn.Module,
) -> float:
    """
    Compute gradient norm across all parameters.

    Args:
        accelerator: Accelerator instance
        model: Model to compute gradient norm for

    Returns:
        Gradient norm value
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def all_reduce_metrics(
    accelerator: Accelerator,
    metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    All-reduce metrics across all processes (for distributed training).

    Args:
        accelerator: Accelerator instance
        metrics: Dictionary of metrics to reduce

    Returns:
        Dictionary with reduced metrics
    """
    if accelerator.num_processes == 1:
        return metrics

    reduced_metrics = {}
    for key, value in metrics.items():
        tensor = torch.tensor(value, device=accelerator.device)
        reduced_tensor = accelerator.reduce(tensor, reduction="mean")
        reduced_metrics[key] = reduced_tensor.item()

    return reduced_metrics


def print_training_info(accelerator: Accelerator, training_config: TrainingConfig):
    """
    Print training configuration and device information.

    Args:
        accelerator: Accelerator instance
        training_config: Training configuration
    """
    accelerator.print("=" * 70)
    accelerator.print("Training Configuration")
    accelerator.print("=" * 70)

    accelerator.print("\nDevice Information:")
    accelerator.print(f"  Device: {accelerator.device}")
    accelerator.print(f"  Num processes: {accelerator.num_processes}")
    accelerator.print(f"  Process index: {accelerator.process_index}")
    accelerator.print(f"  Local process index: {accelerator.local_process_index}")
    accelerator.print(f"  Mixed precision: {accelerator.mixed_precision}")

    accelerator.print("\nTraining Settings:")
    accelerator.print(f"  Epochs: {training_config.epochs}")
    accelerator.print(f"  Batch size: {training_config.batch_size}")
    accelerator.print(
        f"  Gradient accumulation steps: {training_config.gradient_accumulation_steps}"
    )
    accelerator.print(
        f"  Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps * accelerator.num_processes}"
    )

    if training_config.gradient_clipping:
        accelerator.print(f"  Gradient clipping: {training_config.gradient_clipping}")

    accelerator.print(f"\nOptimizer: {training_config.optimizer.type}")
    accelerator.print(f"  Learning rate: {training_config.optimizer.lr}")
    accelerator.print(f"  Weight decay: {training_config.optimizer.weight_decay}")

    accelerator.print(f"\nScheduler: {training_config.scheduler.type}")
    if training_config.scheduler.warmup_steps > 0:
        accelerator.print(f"  Warmup steps: {training_config.scheduler.warmup_steps}")

    if training_config.checkpoint:
        accelerator.print("\nCheckpointing:")
        accelerator.print(f"  Save dir: {training_config.checkpoint.save_dir}")
        accelerator.print(
            f"  Interval: every {training_config.checkpoint.interval} steps"
        )

    if training_config.validation:
        accelerator.print("\nValidation:")
        accelerator.print(
            f"  Interval: every {training_config.validation.interval} steps"
        )

    accelerator.print("\n" + "=" * 70 + "\n")
