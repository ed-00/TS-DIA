#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Optimizer and Scheduler Creation

This module provides utilities for creating optimizers and learning rate
schedulers from configuration.

Key Functions:
    create_optimizer: Create optimizer from config
    create_scheduler: Create LR scheduler from config
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    ReduceLROnPlateau,
    StepLR,
)

from .config import OptimizerConfig, SchedulerConfig


def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: Model to optimize
        config: Optimizer configuration

    Returns:
        PyTorch optimizer

    Example:
        ```python
        optimizer = create_optimizer(model, optimizer_config)
        ```
    """
    optimizer_type = config.type.lower()

    # Base parameters
    params = model.parameters()

    # Create optimizer
    if optimizer_type == "adam":
        return torch.optim.Adam(
            params,
            lr=config.lr,
            betas=config.betas or (0.9, 0.999),
            eps=config.epsilon or 1e-8,
            weight_decay=config.weight_decay or 0.0,
            amsgrad=config.amsgrad or False,
        )

    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=config.betas or (0.9, 0.999),
            eps=config.epsilon or 1e-8,
            weight_decay=config.weight_decay or 0.01,
            amsgrad=config.amsgrad or False,
        )

    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum or 0.0,
            weight_decay=config.weight_decay or 0.0,
            nesterov=config.nesterov or False,
        )

    elif optimizer_type == "adagrad":
        return torch.optim.Adagrad(
            params,
            lr=config.lr,
            eps=config.epsilon or 1e-10,
            weight_decay=config.weight_decay or 0.0,
        )

    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=config.lr,
            momentum=config.momentum or 0.0,
            eps=config.epsilon or 1e-8,
            weight_decay=config.weight_decay or 0.0,
        )

    elif optimizer_type == "adamax":
        return torch.optim.Adamax(
            params,
            lr=config.lr,
            betas=config.betas or (0.9, 0.999),
            eps=config.epsilon or 1e-8,
            weight_decay=config.weight_decay or 0.0,
        )

    else:
        available = ["adam", "adamw", "sgd", "adagrad", "rmsprop", "adamax"]
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. Available: {available}"
        )


def get_warmup_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Create warmup + cosine decay scheduler.

    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum learning rate as ratio of initial LR

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        import math

        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    num_training_steps: int | None = None,
) -> Any:
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        num_training_steps: Total number of training steps (for some schedulers)

    Returns:
        PyTorch LR scheduler

    Example:
        ```python
        scheduler = create_scheduler(optimizer, scheduler_config, num_steps)
        ```
    """
    scheduler_type = config.type.lower()

    if scheduler_type == "cosine":
        if config.warmup_steps > 0 and num_training_steps:
            # Cosine with warmup
            return get_warmup_schedule(
                optimizer,
                warmup_steps=config.warmup_steps,
                total_steps=num_training_steps,
                min_lr_ratio=config.min_lr / optimizer.param_groups[0]["lr"]
                if config.min_lr
                else 0.0,
            )
        else:
            # Simple cosine annealing
            T_max = config.decay_steps or num_training_steps or 1000
            return CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=config.min_lr or 0.0,
            )

    elif scheduler_type == "cosine_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.decay_steps or 1000,
            T_mult=config.num_cycles or 1,
            eta_min=config.min_lr or 0.0,
        )

    elif scheduler_type == "linear":
        if config.warmup_steps > 0:
            # Linear warmup then decay
            def lr_lambda(current_step: int):
                if current_step < config.warmup_steps:
                    return float(current_step) / float(max(1, config.warmup_steps))

                total_decay_steps = (config.decay_steps or num_training_steps or 1000)
                progress = float(current_step - config.warmup_steps) / float(
                    max(1, total_decay_steps - config.warmup_steps)
                )
                return max(0.0, 1.0 - progress)

            return LambdaLR(optimizer, lr_lambda)
        else:
            min_lr_factor = (config.min_lr or 0.0) / optimizer.param_groups[0]["lr"]
            return LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=min_lr_factor,
                total_iters=config.decay_steps or num_training_steps or 1000,
            )

    elif scheduler_type == "exponential":
        return ExponentialLR(optimizer, gamma=config.gamma or 0.95)

    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=config.step_size or 1000,
            gamma=config.gamma or 0.1,
        )

    elif scheduler_type == "plateau":
        # Ensure mode is valid
        mode = "min" if config.mode == "min" else "max"
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=config.gamma or 0.1,
            patience=config.patience or 10,
            min_lr=config.min_lr or 0.0,
        )

    elif scheduler_type == "constant":
        # Constant LR with optional warmup
        if config.warmup_steps > 0:

            def lr_lambda(current_step: int):
                if current_step < config.warmup_steps:
                    return float(current_step) / float(max(1, config.warmup_steps))
                return 1.0

            return LambdaLR(optimizer, lr_lambda)
        else:
            return LambdaLR(optimizer, lambda step: 1.0)

    else:
        available = [
            "cosine",
            "cosine_restarts",
            "linear",
            "exponential",
            "step",
            "plateau",
            "constant",
        ]
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. Available: {available}"
        )
