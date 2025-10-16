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

import random
from typing import Dict, Optional

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
)
from .losses import (
    compute_loss,
    compute_metrics,
    create_auxiliary_losses,
    create_loss_function,
)
from .optimizers import create_optimizer, create_scheduler


class Trainer:
    """
    Universal Trainer for deep learning models.

    Args:
        model: PyTorch model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader (optional)
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
        val_dataloader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
        test_dataloader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # Set random seeds for reproducibility
        self._set_seed(config.random_seed)

        # Setup Accelerate
        self.accelerator = setup_accelerator(
            config,
            project_dir=config.checkpoint.save_dir if config.checkpoint else None,
        )

        # Log system info
        log_system_info(self.accelerator)
        print_training_info(self.accelerator, config)

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(model, config.optimizer)

        # Calculate total training steps
        self.total_steps = self._calculate_total_steps()

        # Guard scheduler against unknown total steps by falling back to decay_steps
        safe_total_steps = self.total_steps or config.scheduler.decay_steps
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
            self.val_dataloader,
            self.lr_scheduler,
        ) = prepare_for_training(
            self.accelerator,
            model,
            self.optimizer,
            train_dataloader,
            val_dataloader,
            self.lr_scheduler,
        )

        # Setup loss functions
        self.loss_fn = (
            create_loss_function(config.loss) if config.loss else nn.CrossEntropyLoss()
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
            )

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

    def _calculate_total_steps(self) -> Optional[int]:
        """Return max_steps if provided; otherwise None (lengthless iteration)."""
        return self.config.max_steps or None

    def _set_sampler_epoch_if_applicable(self, loader: DataLoader, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        self.accelerator.print(f"📂 Resuming from checkpoint: {checkpoint_path}")

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

        # Restore sampler state if available
        sampler_state = extra_state.get("sampler_state")
        if sampler_state is not None and hasattr(self.train_dataloader, "sampler"):
            sampler = getattr(self.train_dataloader, "sampler", None)
            if hasattr(sampler, "load_state_dict"):
                try:
                    sampler.load_state_dict(sampler_state)
                except Exception as e:
                    self.accelerator.print(
                        f"Warning: failed to load sampler state: {e}"
                    )

        self.accelerator.print(
            f"Resumed from epoch {self.current_epoch}, step {self.global_step}"
        )

    def train(self):
        """Main training loop."""
        self.callback_handler.on_train_begin(self)

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.callback_handler.on_epoch_begin(self, epoch)

            # Ensure Lhotse/Distributed samplers advance epoch appropriately
            if self.train_dataloader is not None:
                self._set_sampler_epoch_if_applicable(self.train_dataloader, epoch)
            if self.val_dataloader is not None:
                self._set_sampler_epoch_if_applicable(self.val_dataloader, epoch)

            # Train for one epoch
            train_metrics = self._train_epoch()

            # Validation
            val_metrics = {}
            if self.val_dataloader is not None and self.config.validation:
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

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(progress_bar):
            self.callback_handler.on_batch_begin(self, batch, batch_idx)

            with self.accelerator.accumulate(self.model):
                # Forward pass - extract features from diarization batch
                outputs = self.model(x=batch["features"])

                # Transpose target from [batch, num_speakers, num_frames] to [batch, num_frames, num_speakers]
                # Convert to float32 to match model output dtype
                targets = batch["speaker_activity"].transpose(1, 2).float()

                # Compute loss
                loss_dict = compute_loss(
                    self.loss_fn,
                    outputs if isinstance(outputs, torch.Tensor) else outputs.logits,
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
                progress_bar.set_postfix(loss=metrics["loss"], lr=metrics["lr"])

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
        """Run validation."""
        self.model.eval()
        self.callback_handler.on_validation_begin(self)

        total_loss = 0.0
        total_samples = 0
        all_metrics = []

        # Check if max_steps is set for validation
        max_val_steps = (
            getattr(self.config.validation, "max_steps", None)
            if self.config.validation
            else None
        )

        for batch_idx, batch in enumerate(
            tqdm(
                self.val_dataloader,
                desc="Validation",
                disable=not self.accelerator.is_local_main_process,
            )
        ):
            # Stop if max_steps reached
            if max_val_steps is not None and batch_idx >= max_val_steps:
                break
            outputs = self.model(x=batch["features"])

            # Transpose target from [batch, num_speakers, num_frames] to [batch, num_frames, num_speakers]
            # Convert to float32 to match model output dtype
            targets = batch["speaker_activity"].transpose(1, 2).float()

            # Compute loss
            loss_dict = compute_loss(
                self.loss_fn,
                outputs if isinstance(outputs, torch.Tensor) else outputs.logits,
                targets,
            )

            total_loss += loss_dict["total"].item() * targets.size(0)
            total_samples += targets.size(0)

            # Compute metrics
            batch_metrics = compute_metrics(
                outputs if isinstance(outputs, torch.Tensor) else outputs.logits,
                targets,
                task_type="diarization",
            )
            all_metrics.append(batch_metrics)

        # Average metrics
        val_metrics = {
            "val_loss": total_loss / total_samples,
        }

        # Aggregate batch metrics
        if all_metrics:
            for key in all_metrics[0].keys():
                val_metrics[f"val_{key}"] = np.mean([m[key] for m in all_metrics])

        # All-reduce for distributed
        val_metrics = all_reduce_metrics(self.accelerator, val_metrics)

        # Log metrics via Accelerate
        self.accelerator.log(val_metrics, step=self.global_step)

        self.callback_handler.on_validation_end(self, val_metrics)

        self.accelerator.print(
            f"Validation | Loss: {val_metrics['val_loss']:.4f} | "
            + " | ".join(
                [f"{k}: {v:.4f}" for k, v in val_metrics.items() if k != "val_loss"]
            )
        )

        return val_metrics

    def save_checkpoint(self, name: str, is_best: bool = False):
        """Save training checkpoint."""
        if not self.config.checkpoint:
            return

        extra_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }

        # Save sampler state for exact resumption of data iteration
        if hasattr(self.train_dataloader, "sampler"):
            sampler = getattr(self.train_dataloader, "sampler", None)
            if hasattr(sampler, "state_dict"):
                try:
                    extra_state["sampler_state"] = sampler.state_dict()
                except Exception as e:
                    self.accelerator.print(
                        f"Warning: failed to save sampler state: {e}"
                    )

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

        for batch in tqdm(
            self.test_dataloader,
            desc="Testing",
            disable=not self.accelerator.is_local_main_process,
        ):
            outputs = self.model(x=batch["features"])

            # Transpose target from [batch, num_speakers, num_frames] to [batch, num_frames, num_speakers]
            # Convert to float32 to match model output dtype
            targets = batch["speaker_activity"].transpose(1, 2).float()

            # Compute loss
            loss_dict = compute_loss(
                self.loss_fn,
                outputs if isinstance(outputs, torch.Tensor) else outputs.logits,
                targets,
            )

            total_loss += loss_dict["total"].item() * targets.size(0)
            total_samples += targets.size(0)

            # Compute metrics
            batch_metrics = compute_metrics(
                outputs if isinstance(outputs, torch.Tensor) else outputs.logits,
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
                test_metrics[f"test_{key}"] = np.mean([m[key] for m in all_metrics])

        # All-reduce for distributed
        test_metrics = all_reduce_metrics(self.accelerator, test_metrics)

        # Log metrics via Accelerate
        self.accelerator.log(test_metrics, step=self.global_step)

        self.accelerator.print(
            f"\nTest Results | Loss: {test_metrics['test_loss']:.4f} | "
            + " | ".join(
                [f"{k}: {v:.4f}" for k, v in test_metrics.items() if k != "test_loss"]
            )
        )

        return test_metrics
