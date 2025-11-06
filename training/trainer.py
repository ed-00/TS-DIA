#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Universal Training Loop

This module provides a comprehensive Trainer class that handles:
- Training loop with gradient accumulation
- Mixed precision training via Accelerate
- Distributed training (DDP, FSDP)
- Checkpointing and resuming
- Validation and early stopping
- Logging to multiple backends
- Callback system for extensibility

Key Classes:
    Trainer: Main training class
"""

import numbers
import random
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .accelerate_utils import (
    all_reduce_metrics,
    load_checkpoint_with_accelerate,
    prepare_for_training,
    print_training_info,
    save_checkpoint_with_accelerate,
    setup_accelerator,
)
from .callbacks import CallbackHandler, create_callbacks_from_config
from .config import TrainingConfig
from .logging_utils import (
    init_trackers,
    log_hyperparameters,
    log_model_summary,
    log_system_info,
    save_wandb_info,
    setup_file_logger,
)
from .losses import (
    compute_loss,
    compute_metrics,
    create_auxiliary_losses,
    create_loss_function,
)
from .optimizers import create_optimizer, create_scheduler


class CheckpointDirectoryExistsError(Exception):
    """Exception raised when checkpoint directory exists and resume is not specified."""

    pass


class Trainer:
    """
    Universal Trainer for deep learning models.

    Args:
        model: PyTorch model to train
        train_dataloader: Training data loader
    val_dataloader: Validation data loader or mapping of split names to dataloaders (optional)
        config: Training configuration
        test_dataloader: Test data loader (optional)

    Example:
        ```python
        from training import Trainer, parse_training_config

        config = parse_training_config('config.yml')
        trainer = Trainer(model, train_dl, val_dl, config)
        trainer.train()
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: TrainingConfig,
        val_dataloader: Optional[Union[DataLoader, Dict[str, DataLoader]]] = None,
        test_dataloader: Optional[DataLoader] = None,
        config_path: Optional[str] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config_path = config_path

        # Normalize validation dataloaders input for multi-split evaluation
        self.val_dataloaders: Dict[str, DataLoader] = {}
        self._primary_val_key: Optional[str] = None
        primary_val_dl: Optional[DataLoader] = None

        if isinstance(val_dataloader, dict):
            if val_dataloader:
                self.val_dataloaders = dict(val_dataloader)
                if len(val_dataloader) == 1:
                    self._primary_val_key, primary_val_dl = next(
                        iter(val_dataloader.items())
                    )
        elif val_dataloader is not None:
            self._primary_val_key = "val"
            primary_val_dl = val_dataloader
            self.val_dataloaders[self._primary_val_key] = val_dataloader
        else:
            self.val_dataloaders = {}

        # Set random seeds for reproducibility
        self._set_seed(config.random_seed)

        # Validate checkpoint directory (before creating anything)
        if config.checkpoint:
            self._validate_checkpoint_directory(config.checkpoint)

            # Copy config file to checkpoint directory
            if config_path and Path(config_path).exists():
                self._copy_config_file(config.checkpoint.save_dir, config_path)

        # Setup Accelerate
        self.accelerator = setup_accelerator(
            config,
            project_dir=config.checkpoint.save_dir if config.checkpoint else None,
        )

        # Setup file logging
        if config.checkpoint:
            is_resume = config.checkpoint.resume is not None
            log_file = setup_file_logger(
                config.checkpoint.save_dir, is_resume=is_resume
            )
            self.accelerator.print(f"Console output logging to: {log_file}")

        # Log system info
        log_system_info(self.accelerator)
        print_training_info(self.accelerator, config)

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(model, config.optimizer)

        # Calculate total training steps
        self.total_steps = self._calculate_total_steps()

        # Estimate steps per epoch using dataloader length and gradient accumulation
        estimated_batches_per_epoch = len(train_dataloader)
        if estimated_batches_per_epoch is not None and not self.total_steps:
            # Calculate steps per epoch considering gradient accumulation
            steps_per_epoch = estimated_batches_per_epoch // config.gradient_accumulation_steps
            estimated_total_steps = steps_per_epoch * config.epochs
            self.accelerator.print(
                f"Estimated training steps: {estimated_total_steps} "
                f"({estimated_batches_per_epoch} batches/epoch ÷ {config.gradient_accumulation_steps} grad_accum × {config.epochs} epochs)"
            )
        else:
            estimated_total_steps = None

        # Use estimated steps, explicit total_steps, or fallback to decay_steps
        safe_total_steps = (
            self.total_steps or
            estimated_total_steps or
            config.scheduler.decay_steps or
            1000  # Final fallback
        )

        self.lr_scheduler = create_scheduler(
            self.optimizer,
            config.scheduler,
            num_training_steps=safe_total_steps,
        )

        # Prepare for training with Accelerate
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            prepared_primary_val_dl,
            self.lr_scheduler,
        ) = prepare_for_training(
            self.accelerator,
            model,
            self.optimizer,
            train_dataloader,
            primary_val_dl,
            self.lr_scheduler,
        )

        # Ensure validation dataloaders are prepared for distributed execution
        if self.val_dataloaders:
            if self._primary_val_key and prepared_primary_val_dl is not None:
                self.val_dataloaders[self._primary_val_key] = prepared_primary_val_dl

            if len(self.val_dataloaders) > 1 or prepared_primary_val_dl is None:
                for split_name, dataloader in list(self.val_dataloaders.items()):
                    if (
                        self._primary_val_key == split_name
                        and prepared_primary_val_dl is not None
                    ):
                        continue
                    self.val_dataloaders[split_name] = self.accelerator.prepare_data_loader(
                        dataloader
                    )

        # Maintain backward compatibility attribute
        self.val_dataloader = (
            self.val_dataloaders[self._primary_val_key]
            if self._primary_val_key and self._primary_val_key in self.val_dataloaders
            else None
        )

        # Setup loss functions
        self.loss_fn = (
            create_loss_function(
                config.loss) if config.loss else nn.CrossEntropyLoss()
        )
        self.auxiliary_losses = (
            create_auxiliary_losses(config.loss) if config.loss else {}
        )
        self.auxiliary_weights = config.loss.auxiliary if config.loss else {}

        # Setup logging via Accelerate
        if config.logging and (config.logging.tensorboard or config.logging.wandb):
            init_trackers(
                self.accelerator,
                config.logging,
                config,
                project_name=config.logging.wandb_project or "training",
                config_path=config_path,
            )

            # Save wandb info if wandb is enabled
            if config.logging.wandb and config.checkpoint:
                save_wandb_info(self.accelerator, config.checkpoint.save_dir)

        # Log hyperparameters
        if config.checkpoint:
            log_hyperparameters(
                self.accelerator,
                {
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "optimizer": config.optimizer.type,
                    "lr": config.optimizer.lr,
                    "scheduler": config.scheduler.type,
                },
                save_dir=config.checkpoint.save_dir,
            )

        # Log model summary
        if config.logging and config.logging.log_model and config.checkpoint:
            log_model_summary(
                self.accelerator,
                self.model,
                save_dir=config.checkpoint.save_dir,
            )

        # Setup callbacks
        self.callback_handler = CallbackHandler(
            create_callbacks_from_config(config, self.accelerator)
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")

        # Resume from checkpoint if specified
        if config.checkpoint and config.checkpoint.resume:
            self._resume_from_checkpoint(config.checkpoint.resume)

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _validate_checkpoint_directory(self, checkpoint_config):
        """
        Validate checkpoint directory doesn't exist unless resuming.

        Args:
            checkpoint_config: CheckpointConfig object

        Raises:
            CheckpointDirectoryExistsError: If directory exists and not resuming
        """
        checkpoint_dir = Path(checkpoint_config.save_dir)

        # If resuming, directory should exist - no validation needed
        if checkpoint_config.resume:
            return

        # If not resuming, directory should NOT exist
        if checkpoint_dir.exists():
            raise CheckpointDirectoryExistsError(
                f"\n{'=' * 70}\n"
                f"ERROR: Checkpoint directory already exists!\n"
                f"{'=' * 70}\n"
                f"Directory: {checkpoint_dir}\n\n"
                f"To prevent accidental overwriting of existing experiments, training cannot proceed.\n\n"
                f"Please choose ONE of the following options:\n"
                f"  1. Change 'checkpoint.save_dir' in your config to a new path\n"
                f"  2. Set 'checkpoint.resume' to resume from this checkpoint\n"
                f"  3. Manually delete the directory if you want to overwrite it\n"
                f"{'=' * 70}\n"
            )

    def _copy_config_file(self, checkpoint_dir: str, config_path: str):
        """
        Copy configuration file to checkpoint directory.

        Args:
            checkpoint_dir: Checkpoint directory path
            config_path: Path to configuration file
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        config_dest = checkpoint_path / "config.yml"
        shutil.copy2(config_path, config_dest)

        print(f"Configuration file copied to: {config_dest}")

    def _calculate_total_steps(self) -> Optional[int]:
        """Return max_steps if provided; otherwise None (lengthless iteration)."""
        return self.config.max_steps or None

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        self.accelerator.print(
            f"📂 Resuming from checkpoint: {checkpoint_path}")

        extra_state = load_checkpoint_with_accelerate(
            self.accelerator,
            checkpoint_path,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        )

        # Restore training state
        self.current_epoch = extra_state.get("epoch", 0)
        self.global_step = extra_state.get("global_step", 0)
        self.best_metric = extra_state.get("best_metric", float("inf"))

        self.accelerator.print(
            f"Resumed from epoch {self.current_epoch}, step {self.global_step}"
        )

    def train(self):
        """Main training loop."""
        self.callback_handler.on_train_begin(self)

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.callback_handler.on_epoch_begin(self, epoch)

            # Train for one epoch
            train_metrics = self._train_epoch()

            # Validation
            val_metrics = {}
            if self.val_dataloaders and self.config.validation:
                if (epoch + 1) % self.config.validation.interval == 0 or epoch == 0:
                    val_metrics = self.validate()

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Callbacks
            self.callback_handler.on_epoch_end(self, epoch, epoch_metrics)

            # Check early stopping
            if self.callback_handler.should_stop_training():
                self.accelerator.print("Early stopping triggered")
                break

            # Save checkpoint
            if (
                self.config.checkpoint
                and (epoch + 1) % self.config.checkpoint.interval == 0
            ):
                self.save_checkpoint(f"epoch_{epoch + 1}")

        self.callback_handler.on_train_end(self)

        # End tracking
        self.accelerator.end_training()

        self.accelerator.print("Training completed!")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        # Estimate total batches for tqdm progress bar
        total_batches = len(self.train_dataloader)
        self.accelerator.print(
            f"Training for {total_batches} batches per epoch")

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
            disable=not self.accelerator.is_local_main_process,
            total=total_batches,
        )

        for batch_idx, batch in enumerate(progress_bar):
            self.callback_handler.on_batch_begin(self, batch, batch_idx)

            with self.accelerator.accumulate(self.model):
                # Ego-centric diarization: labels are [batch, num_frames] with class indices
                targets = batch["labels"]

                # Forward pass - extract features from diarization batch
                outputs = self.model(x=batch["features"])

                # Compute loss
                loss_dict = compute_loss(
                    self.loss_fn,
                    outputs if isinstance(
                        outputs, torch.Tensor) else outputs.logits,
                    targets,
                    auxiliary_losses=self.auxiliary_losses,
                    auxiliary_weights=self.auxiliary_weights,
                    model=self.model,
                )

                loss = loss_dict["total"]

                # Backward pass
                self.accelerator.backward(loss)

                self.callback_handler.on_backward_end(self)

                # Gradient clipping (if not using callback)
                if self.config.gradient_clipping and not any(
                    hasattr(cb, "max_norm") for cb in self.callback_handler.callbacks
                ):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clipping,
                    )

                # Optimizer step
                self.optimizer.step()

                # Scheduler step
                if self.lr_scheduler and not isinstance(
                    self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.lr_scheduler.step()

                self.callback_handler.on_optimizer_step(self)

                self.optimizer.zero_grad()

            # Logging
            total_loss += loss.item()
            num_batches += 1

            if (
                self.config.logging
                and self.global_step % self.config.logging.interval == 0
            ):
                # Compute metrics
                metrics = {
                    "loss": loss.item(),
                    "main_loss": loss_dict["main"].item(),
                }

                # Add auxiliary losses
                for aux_name, aux_loss in loss_dict.items():
                    if aux_name not in ["total", "main"]:
                        metrics[f"{aux_name}_loss"] = aux_loss.item()

                # Add learning rate
                metrics["lr"] = self.optimizer.param_groups[0]["lr"]

                # All-reduce metrics for distributed training
                metrics = all_reduce_metrics(self.accelerator, metrics)

                # Log to backends via Accelerate
                self.accelerator.log(metrics, step=self.global_step)

                # Update progress bar
                progress_bar.set_postfix(
                    loss=metrics["loss"], lr=metrics["lr"])

            self.callback_handler.on_batch_end(self, batch, batch_idx, outputs)
            self.global_step += 1

            # Save checkpoint at step intervals
            if (
                self.config.checkpoint
                and self.global_step % self.config.checkpoint.interval == 0
            ):
                self.save_checkpoint(f"step_{self.global_step}")

            # Check if we've reached max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation across all configured splits."""
        if not self.val_dataloaders:
            return {}

        self.model.eval()
        self.callback_handler.on_validation_begin(self)

        max_val_steps = (
            getattr(self.config.validation, "max_steps", None)
            if self.config.validation
            else None
        )

        split_metrics: Dict[str, Dict[str, float]] = {}

        for split_name, dataloader in self.val_dataloaders.items():
            total_loss = 0.0
            total_samples = 0
            all_metrics = []

            try:
                total_batches = len(dataloader)
            except TypeError:
                total_batches = None

            for batch_idx, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Validation[{split_name}]",
                    disable=not self.accelerator.is_local_main_process,
                    total=total_batches,
                )
            ):
                if max_val_steps is not None and batch_idx >= max_val_steps:
                    break

                outputs = self.model(x=batch["features"])

                if "labels" in batch:
                    targets = batch["labels"]
                elif "speaker_activity" in batch:
                    targets = batch["speaker_activity"].transpose(1, 2).float()
                else:
                    raise ValueError(
                        "Batch must contain either 'labels' or 'speaker_activity'"
                    )

                loss_dict = compute_loss(
                    self.loss_fn,
                    outputs if isinstance(outputs, torch.Tensor) else outputs.logits,
                    targets,
                )

                total_loss += loss_dict["total"].item() * targets.size(0)
                total_samples += targets.size(0)

                batch_metrics = compute_metrics(
                    outputs if isinstance(outputs, torch.Tensor) else outputs.logits,
                    targets,
                    task_type="diarization",
                )
                all_metrics.append(batch_metrics)

            if total_samples == 0:
                split_results = {"val_loss": float("nan")}
            else:
                split_results = {"val_loss": total_loss / total_samples}

            if all_metrics:
                for key in all_metrics[0].keys():
                    split_results[f"val_{key}"] = float(
                        np.mean([m[key] for m in all_metrics])
                    )

            split_results = all_reduce_metrics(self.accelerator, split_results)
            split_metrics[split_name] = split_results

            metrics_str = " | ".join(
                f"{metric}: {value:.4f}" for metric, value in split_results.items()
            )
            self.accelerator.print(f"Validation[{split_name}] | {metrics_str}")

        aggregated_metrics = self._aggregate_validation_metrics(split_metrics)

        if self.config.logging and (
            self.config.logging.tensorboard or self.config.logging.wandb
        ):
            log_payload = {
                f"{split}/{metric}": value
                for split, metrics in split_metrics.items()
                for metric, value in metrics.items()
            }
            log_payload.update(aggregated_metrics)
            self.accelerator.log(log_payload, step=self.global_step)
            self.accelerator.print(
                "✓ Validation metrics logged to trackers for splits: "
                + ", ".join(split_metrics.keys())
            )

        self.callback_handler.on_validation_end(self, aggregated_metrics)

        if aggregated_metrics:
            metric_summary = " | ".join(
                f"{metric}: {value:.4f}" for metric, value in aggregated_metrics.items()
            )
            self.accelerator.print(f"Validation (avg) | {metric_summary}")

        return aggregated_metrics

    def _aggregate_validation_metrics(
        self, split_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Average metrics across validation splits."""
        if not split_metrics:
            return {}

        numeric_keys: Optional[set[str]] = None
        for metrics in split_metrics.values():
            current_keys = {
                key for key, value in metrics.items() if isinstance(value, numbers.Number)
            }
            if numeric_keys is None:
                numeric_keys = current_keys
            else:
                numeric_keys &= current_keys

        if not numeric_keys:
            return {}

        aggregated: Dict[str, float] = {}
        for key in numeric_keys:
            aggregated[key] = float(
                np.mean([metrics[key] for metrics in split_metrics.values()])
            )

        return aggregated

    def save_checkpoint(self, name: str, is_best: bool = False):
        """Save training checkpoint."""
        if not self.config.checkpoint:
            return

        extra_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }

        save_checkpoint_with_accelerate(
            self.accelerator,
            self.config.checkpoint.save_dir,
            epoch=self.current_epoch,
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            extra_state=extra_state,
            is_best=is_best,
        )

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """Run final test evaluation."""
        if not self.test_dataloader:
            self.accelerator.print("WARNING: No test dataloader provided")
            return {}

        self.accelerator.print("\nRunning final test evaluation...")

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_metrics = []

        # Estimate total batches for tqdm
        total_batches = len(self.test_dataloader)

        for batch in tqdm(
            self.test_dataloader,
            desc="Testing",
            disable=not self.accelerator.is_local_main_process,
            total=total_batches,
        ):

            outputs = self.model(x=batch["features"])

            # Handle different label formats (ego-centric vs binary diarization)
            if "labels" in batch:
                # Ego-centric diarization: labels are [batch, num_frames] with class indices
                targets = batch["labels"]
            elif "speaker_activity" in batch:
                # Binary diarization: transpose from [batch, num_speakers, num_frames] to [batch, num_frames, num_speakers]
                targets = batch["speaker_activity"].transpose(1, 2).float()
            else:
                raise ValueError(
                    "Batch must contain either 'labels' or 'speaker_activity'")

            # Compute loss
            loss_dict = compute_loss(
                self.loss_fn,
                outputs if isinstance(
                    outputs, torch.Tensor) else outputs.logits,
                targets,
            )

            total_loss += loss_dict["total"].item() * targets.size(0)
            total_samples += targets.size(0)

            # Compute metrics
            batch_metrics = compute_metrics(
                outputs if isinstance(
                    outputs, torch.Tensor) else outputs.logits,
                targets,
                task_type="diarization",
            )
            all_metrics.append(batch_metrics)

        # Average metrics
        test_metrics = {
            "test_loss": total_loss / total_samples,
        }

        if all_metrics:
            for key in all_metrics[0].keys():
                test_metrics[f"test_{key}"] = np.mean(
                    [m[key] for m in all_metrics])

        # All-reduce for distributed
        test_metrics = all_reduce_metrics(self.accelerator, test_metrics)

        # Log metrics via Accelerate
        self.accelerator.log(test_metrics, step=self.global_step)

        self.accelerator.print(
            f"\nTest Results | Loss: {test_metrics['test_loss']:.4f} | "
            + " | ".join(
                [f"{k}: {v:.4f}" for k, v in test_metrics.items() if k !=
                 "test_loss"]
            )
        )

        return test_metrics
