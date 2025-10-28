#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Training Callbacks System

This module provides an extensible callback system for training hooks including:
- Early stopping
- Feature redrawing for Performer models
- Learning rate monitoring
- Custom callbacks

Key Classes:
    Callback: Base callback class
    EarlyStoppingCallback: Early stopping based on metric
    FeatureRedrawCallback: Redraw Performer feature matrices
    LRMonitorCallback: Monitor learning rate changes
    CallbackHandler: Manages multiple callbacks
"""

from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .config import EarlyStoppingConfig


class Callback(ABC):
    """
    Base class for training callbacks.

    Callbacks can hook into various training events:
    - on_train_begin/end
    - on_epoch_begin/end
    - on_batch_begin/end
    - on_validation_begin/end
    """

    def on_train_begin(self, trainer: Any):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: Any):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer: Any, batch: Any, batch_idx: int):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, trainer: Any, batch: Any, batch_idx: int, outputs: Any):
        """Called at the end of each batch."""
        pass

    def on_validation_begin(self, trainer: Any):
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]):
        """Called at the end of validation."""
        pass

    def on_backward_end(self, trainer: Any):
        """Called after backward pass."""
        pass

    def on_optimizer_step(self, trainer: Any):
        """Called after optimizer step."""
        pass


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback to stop training when metric stops improving.

    Args:
        config: Early stopping configuration
        accelerator: Accelerator instance for logging
    """

    def __init__(self, config: EarlyStoppingConfig, accelerator: Any = None):
        self.config = config
        self.accelerator = accelerator
        self.best_metric = float("inf") if config.mode == "min" else float("-inf")
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def _is_improvement(self, current_metric: float) -> bool:
        """Check if current metric is an improvement."""
        if self.config.mode == "min":
            return current_metric < (self.best_metric - self.config.min_delta)
        else:
            return current_metric > (self.best_metric + self.config.min_delta)

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]):
        """Check early stopping criteria after validation."""
        if self.config.metric not in metrics:
            if self.accelerator:
                self.accelerator.print(
                    f"Warning: Early stopping metric '{self.config.metric}' not found in metrics. "
                    f"Available: {list(metrics.keys())}"
                )
            return

        current_metric = metrics[self.config.metric]

        if self._is_improvement(current_metric):
            self.best_metric = current_metric
            self.counter = 0
            self.best_epoch = trainer.current_epoch

            if self.accelerator:
                self.accelerator.print(
                    f"New best {self.config.metric}: {current_metric:.6f}"
                )
        else:
            self.counter += 1

            if self.accelerator:
                self.accelerator.print(
                    f"EarlyStopping counter: {self.counter}/{self.config.patience}"
                )

            if self.counter >= self.config.patience:
                self.should_stop = True
                if self.accelerator:
                    self.accelerator.print(
                        f"Early stopping triggered! Best {self.config.metric}: {self.best_metric:.6f} "
                        f"at epoch {self.best_epoch}"
                    )


class FeatureRedrawCallback(Callback):
    """
    Callback to redraw Performer feature matrices at specified intervals.

    This is specific to Performer/linear attention models and helps
    maintain approximation quality during training.

    Args:
        interval: Redraw features every N steps
        fixed_projection: If True, use fixed random projection
    """

    def __init__(self, interval: int, fixed_projection: bool = False):
        self.interval = interval
        self.fixed_projection = fixed_projection
        self.step_count = 0

    def on_batch_end(self, trainer: Any, batch: Any, batch_idx: int, outputs: Any):
        """Redraw features if interval reached."""
        self.step_count += 1

        if self.interval and self.step_count % self.interval == 0:
            if not self.fixed_projection:
                self._redraw_features(trainer.model)

    def _redraw_features(self, model: nn.Module):
        """Redraw projection features for Performer attention layers."""
        for module in model.modules():
            if hasattr(module, "redraw_projection_matrix"):
                module.redraw_projection_matrix()


class LRMonitorCallback(Callback):
    """
    Callback to monitor and log learning rate changes.
    """

    def __init__(self, accelerator: Any = None):
        self.accelerator = accelerator
        self.last_lr = None

    def on_optimizer_step(self, trainer: Any):
        """Log learning rate after optimizer step."""
        if trainer.lr_scheduler is not None:
            current_lr = trainer.optimizer.param_groups[0]["lr"]

            if self.last_lr is None or abs(current_lr - self.last_lr) > 1e-10:
                if self.accelerator:
                    self.accelerator.print(f"Learning rate: {current_lr:.2e}")
                self.last_lr = current_lr


class GradientClippingCallback(Callback):
    """
    Callback to apply gradient clipping.

    Args:
        max_norm: Maximum gradient norm
        norm_type: Type of norm (default: 2)
    """

    def __init__(self, max_norm: float, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_backward_end(self, trainer: Any):
        """Apply gradient clipping after backward pass."""
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), self.max_norm, norm_type=self.norm_type
            )


class ModelCheckpointCallback(Callback):
    """
    Callback to save model checkpoints.

    Args:
        save_dir: Directory to save checkpoints
        interval: Save every N steps
        save_best_only: Only save when metric improves
        monitor_metric: Metric to monitor for best model
        mode: 'min' or 'max' for metric
    """

    def __init__(
        self,
        save_dir: str,
        interval: int,
        save_best_only: bool = False,
        monitor_metric: Optional[str] = None,
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.step_count = 0

    def _is_improvement(self, current_metric: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric

    def on_batch_end(self, trainer: Any, batch: Any, batch_idx: int, outputs: Any):
        """Save checkpoint at intervals."""
        self.step_count += 1

        if self.step_count % self.interval == 0:
            if not self.save_best_only:
                self._save_checkpoint(trainer, f"step_{self.step_count}")

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]):
        """Save best checkpoint after validation."""
        if self.save_best_only and self.monitor_metric:
            if self.monitor_metric in metrics:
                current_metric = metrics[self.monitor_metric]

                if self._is_improvement(current_metric):
                    self.best_metric = current_metric
                    self._save_checkpoint(trainer, "best", is_best=True)

    def _save_checkpoint(self, trainer: Any, name: str, is_best: bool = False):
        """Save checkpoint to disk."""
        # This will be implemented by the trainer
        if hasattr(trainer, "save_checkpoint"):
            trainer.save_checkpoint(name, is_best=is_best)


class CallbackHandler:
    """
    Manages multiple callbacks and dispatches events.

    Args:
        callbacks: List of callback instances
    """

    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: Callback):
        """Add a callback to the handler."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callback):
        """Remove a callback from the handler."""
        self.callbacks.remove(callback)

    def on_train_begin(self, trainer: Any):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: Any):
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, trainer: Any, epoch: int):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics)

    def on_batch_begin(self, trainer: Any, batch: Any, batch_idx: int):
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch, batch_idx)

    def on_batch_end(self, trainer: Any, batch: Any, batch_idx: int, outputs: Any):
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch, batch_idx, outputs)

    def on_validation_begin(self, trainer: Any):
        for callback in self.callbacks:
            callback.on_validation_begin(trainer)

    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]):
        for callback in self.callbacks:
            callback.on_validation_end(trainer, metrics)

    def on_backward_end(self, trainer: Any):
        for callback in self.callbacks:
            callback.on_backward_end(trainer)

    def on_optimizer_step(self, trainer: Any):
        for callback in self.callbacks:
            callback.on_optimizer_step(trainer)

    def should_stop_training(self) -> bool:
        """Check if any callback requests training to stop."""
        for callback in self.callbacks:
            if hasattr(callback, "should_stop") and callback.should_stop:
                return True
        return False


def create_callbacks_from_config(
    training_config: Any,
    accelerator: Any = None,
) -> List[Callback]:
    """
    Create callbacks from training configuration.

    Args:
        training_config: Training configuration
        accelerator: Accelerator instance

    Returns:
        List of callback instances
    """
    callbacks = []

    # Early stopping
    if training_config.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(training_config.early_stopping, accelerator)
        )

    # Feature redraw for Performer
    if training_config.feature_redraw_interval:
        callbacks.append(
            FeatureRedrawCallback(
                training_config.feature_redraw_interval,
                training_config.fixed_projection,
            )
        )

    # Gradient clipping (if not using Accelerate's built-in)
    if training_config.gradient_clipping:
        callbacks.append(GradientClippingCallback(training_config.gradient_clipping))

    # Learning rate monitoring
    callbacks.append(LRMonitorCallback(accelerator))

    # Custom callbacks from config
    for callback_name in training_config.callbacks:
        if callback_name == "pruning":
            # Add pruning callback if implemented
            pass
        elif callback_name == "freeze_layers":
            # Add layer freezing callback if implemented
            pass
        elif callback_name == "dynamic_lr":
            # Add dynamic LR callback if implemented
            pass

    return callbacks
