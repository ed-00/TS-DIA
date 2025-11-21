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
from typing import Dict, Optional, Union, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

from .accelerate_utils import (
    all_reduce_metrics,
    prepare_for_training,
    print_training_info,
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
        train_dataloader: Tuple[DataLoader, int],
        config: TrainingConfig,
        accelerator: Accelerator,
        val_dataloader: Optional[Union[DataLoader,
                                       Dict[str, Tuple[DataLoader, int]]]] = None,
        test_dataloader: Optional[DataLoader] = None,
        config_path: Optional[str] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader[0]
        self.total_training_size = math.ceil(train_dataloader[1] / 2)
        self.test_dataloader = test_dataloader
        self.config_path = config_path

        # Normalize validation dataloaders input for multi-split evaluation
        self.val_dataloaders: Dict[str, DataLoader] = {}
        self.val_total_sizes: Dict[str, int] = {}
        self._primary_val_key: Optional[str] = None
        primary_val_dl: Optional[DataLoader] = None

        if isinstance(val_dataloader, dict):
            if val_dataloader:
                self.val_dataloaders = {k: v[0]
                                        for k, v in val_dataloader.items()}
                if len(val_dataloader) == 1:
                    self._primary_val_key, (primary_val_dl, _) = next(
                        iter(val_dataloader.items())
                    )
                for k, v in val_dataloader.items():
                    self.val_total_sizes[k] = v[1]
        elif val_dataloader is not None:
            self._primary_val_key = "val"
            primary_val_dl = val_dataloader
            self.val_dataloaders[self._primary_val_key] = val_dataloader
            # If val_dataloader is a tuple (DataLoader, int), unpack accordingly
            if isinstance(val_dataloader, tuple) and len(val_dataloader) == 2:
                self.val_total_sizes[self._primary_val_key] = val_dataloader[1]
        else:
            self.val_dataloaders = {}

        # Set random seeds for reproducibility
        self._set_seed(config.random_seed)

        # Use provided accelerator or create new one
        if accelerator is not None:
            self.accelerator = accelerator
        else:
            # Only create accelerator if not provided (backward compatibility)
            self.accelerator = setup_accelerator(
                config,
                project_dir=config.checkpoint.save_dir if config.checkpoint else None,
            )

            # Setup file logging (only if we created the accelerator)

            if config.checkpoint:
                is_resume = config.checkpoint.resume is not None
                log_file = setup_file_logger(
                    config.checkpoint.save_dir, is_resume=is_resume
                )
                self.accelerator.print(
                    f"Console output logging to: {log_file}")

            # Log system info
            log_system_info(self.accelerator)
            print_training_info(self.accelerator, config)

        # Copy config file to checkpoint directory (only on main process)
        if self.accelerator.is_main_process:
            if (
                config.checkpoint is not None
                and config.checkpoint.save_dir
                and config_path
                and Path(config_path).exists()
            ):
                self._copy_config_file(
                    config.checkpoint.save_dir, config_path)

        # Create optimizer and scheduler
        self.optimizer = create_optimizer(model, config.optimizer)

        # Calculate training steps and sizes
        self.global_training_examples = int(self.total_training_size) 
        self.global_steps_per_epoch = math.ceil(
            self.global_training_examples
            / (config.gradient_accumulation_steps * config.batch_size)
        )
        self.examples_per_process = math.ceil(
            self.global_training_examples /
            max(1, self.accelerator.num_processes)
        )
        self.steps_per_gpu_expected = math.ceil(
            self.examples_per_process /
            (config.gradient_accumulation_steps * config.batch_size)
        )

        # Compatibility variable used to set scheduler total steps
        self.safe_total_steps = self.global_steps_per_epoch
        num_scheduler_steps =  self.safe_total_steps * config.epochs
        accelerator.print("="*60)
        accelerator.print("Global training examples:", self.global_training_examples)
        accelerator.print("Global steps per epoch:", self.global_steps_per_epoch)
        accelerator.print("Examples per process:", self.examples_per_process)
        accelerator.print("Expected steps per process:", self.steps_per_gpu_expected)
        accelerator.print("="*60)

        if config.scheduler is not None:
            self.lr_scheduler = create_scheduler(
                self.optimizer,
                config.scheduler,
                num_training_steps=num_scheduler_steps,
            )
        else:
            self.lr_scheduler = None
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
            self.train_dataloader,
            primary_val_dl,
            self.lr_scheduler,
        )

        # Register scheduler for checkpointing with Accelerate
        if self.lr_scheduler:
            self.accelerator.register_for_checkpointing(self.lr_scheduler)

        # Register the trainer itself for checkpointing to save/load training state
        self.accelerator.register_for_checkpointing(self)

        # Ensure validation dataloaders are prepared for distributed execution
        if self.val_dataloaders:
            if self._primary_val_key and prepared_primary_val_dl is not None:
                self.val_dataloaders[self._primary_val_key] = prepared_primary_val_dl

            # Prepare any remaining validation dataloaders that were not part of the initial prepare call
            # This is crucial for multi-split validation to avoid deadlocks.
            for split_name, dataloader in self.val_dataloaders.items():
                # Skip the one that's already prepared
                if split_name == self._primary_val_key and prepared_primary_val_dl is not None:
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
                    **({"scheduler": config.scheduler.type} if config.scheduler else {}),
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
        # After data loaders have been prepared we can compute debug info about
        # per-worker expected sizes (useful for IterableDataset sharding)
        try:
            num_workers = getattr(self.train_dataloader, "num_workers", 1)
            self.examples_per_worker = math.ceil(
                self.global_training_examples /
                max(1, self.accelerator.num_processes * max(1, num_workers))
            )
            self.train_dataloader_num_workers = num_workers
        except Exception:
            self.examples_per_worker = None
            self.train_dataloader_num_workers = None
        if config.checkpoint and config.checkpoint.resume:
            self._resume_from_checkpoint(config.checkpoint.resume)

        # Optional anomaly detection enabled via environment for debugging
        import os

        if os.environ.get("DETECT_ANOMALY", "0").lower() in ("1", "true", "yes"):
            # Only enable for the main process to avoid verbose output.
            if self.accelerator.is_local_main_process:
                torch.autograd.set_detect_anomaly(True)

    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict["current_epoch"]
        self.global_step = state_dict["global_step"]
        self.best_metric = state_dict["best_metric"]
        self.accelerator.print(
            f"Restored trainer state: epoch {self.current_epoch}, step {self.global_step}"
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

        self.accelerator.print(f"Configuration file copied to: {config_dest}")

    def _calculate_total_steps(self) -> Optional[int]:
        """Return max_steps if provided; otherwise None (lengthless iteration)."""
        return self.config.max_steps or None

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        self.accelerator.print(
            f"📂 Resuming from checkpoint: {checkpoint_path}")
        self.accelerator.load_state(checkpoint_path)

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
                self.save_checkpoint()

        self.callback_handler.on_train_end(self)

        # Save final checkpoint
        if self.config.checkpoint:
            self.save_checkpoint()

        # End tracking
        self.accelerator.end_training()

        self.accelerator.print("Training completed!")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        self.accelerator.print(
            f"Training for {self.global_steps_per_epoch} global steps per epoch "
            f"({self.global_training_examples} examples),\n  -> {self.examples_per_process} examples per process, "
            f"{self.steps_per_gpu_expected} batches per process (batch_size={self.config.batch_size}, grad_accum={self.config.gradient_accumulation_steps})"
        )
        if getattr(self, 'examples_per_worker', None) is not None:
            self.accelerator.print(
                f"  -> {self.examples_per_worker} examples per worker ({self.train_dataloader_num_workers} workers per process)"
            )

        steps_per_gpu = self.steps_per_gpu_expected
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
            disable=not self.accelerator.is_local_main_process,
            total=steps_per_gpu,
        )

        for batch_idx, batch in enumerate(progress_bar):
            self.callback_handler.on_batch_begin(self, batch, batch_idx)
            with self.accelerator.accumulate(self.model):
                # Ego-centric diarization: labels are [batch, num_frames] with class indices
                targets = batch["labels"]

                # Basic sanity check for batch data
                try:
                    features_ok = torch.isfinite(batch["features"]).all()
                except Exception:
                    features_ok = False

                try:
                    targets_ok = torch.isfinite(targets).all()
                except Exception:
                    targets_ok = False

                if not features_ok or not targets_ok:
                    self.accelerator.print(
                        f"⚠️ Non-finite values found in inputs (features_ok={features_ok}, targets_ok={targets_ok}). Skipping batch {batch_idx} at epoch {self.current_epoch}."
                    )
                    # Zero gradients to avoid contamination
                    try:
                        self.optimizer.zero_grad(set_to_none=True)
                    except Exception:
                        self.optimizer.zero_grad()
                    continue

                # Forward pass - extract features from diarization batch
                outputs = self.model(
                    x=batch["features"],
                    is_target=batch.get("is_target"),
                    labels=batch["labels"],
                )

                try:
                    logits = (
                        outputs
                        if isinstance(outputs, torch.Tensor)
                        else outputs.logits
                    )
                except Exception:
                    logits = None

                if logits is not None:
                    if not torch.isfinite(logits).all():
                        # Save the problematic batch and some statistics for debugging
                        diag_dir = None
                        try:
                            save_root = None
                            if self.config.checkpoint and self.config.checkpoint.save_dir:
                                save_root = self.config.checkpoint.save_dir
                            if save_root:
                                diag_dir = Path(save_root) / "diag_nan_batches"
                                diag_dir.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            diag_dir = None

                        try:
                            if diag_dir is not None:
                                torch.save(
                                    {
                                        "epoch": int(self.current_epoch),
                                        "step": int(self.global_step),
                                        "batch_idx": int(batch_idx),
                                        "features": batch.get("features"),
                                        "labels": batch.get("labels"),
                                        "logits_min": float(torch.min(logits).cpu().item()),
                                        "logits_max": float(torch.max(logits).cpu().item()),
                                    },
                                    diag_dir /
                                    f"bad_batch_epoch{self.current_epoch}_step{self.global_step}_batch{batch_idx}.pt",
                                )
                        except Exception:
                            pass

                        self.accelerator.print(
                            f"⚠️ Non-finite logits detected at epoch {self.current_epoch}, step {self.global_step}, batch {batch_idx}. Saving diagnostics and raising RuntimeError."
                        )
                        # Zero out gradients to avoid contaminating optimizer state, then continue
                        try:
                            # Save diagnostics
                            save_root = None
                            if self.config.checkpoint and self.config.checkpoint.save_dir:
                                save_root = self.config.checkpoint.save_dir
                            if save_root is None:
                                raise RuntimeError(
                                    f"Non-finite logits detected at epoch {self.current_epoch}, step {self.global_step}, batch {batch_idx}. No checkpoint save_dir available to store diagnostics."
                                )
                            diag_dir = Path(save_root) / "diag_nan_batches"
                            diag_dir.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                {
                                    "epoch": int(self.current_epoch),
                                    "step": int(self.global_step),
                                    "batch_idx": int(batch_idx),
                                    "features": batch.get("features"),
                                    "labels": batch.get("labels"),
                                    "logits_min": float(torch.min(logits).cpu().item()),
                                    "logits_max": float(torch.max(logits).cpu().item()),
                                },
                                diag_dir /
                                f"bad_batch_epoch{self.current_epoch}_step{self.global_step}_batch{batch_idx}.pt",
                            )
                        except Exception as e:
                            self.accelerator.print(
                                f"Failed saving logits diagnostics: {e}")
                        raise RuntimeError(
                            f"Non-finite logits detected at epoch {self.current_epoch}, step {self.global_step}, batch {batch_idx}. See diag files if saved."
                        )

                # Compute loss, ignoring the first token (enrollment)
                loss_dict = compute_loss(
                    self.loss_fn,
                    outputs[:, 1:, :]
                    if isinstance(outputs, torch.Tensor)
                    else outputs.logits[:, 1:, :],
                    targets,
                    auxiliary_losses=self.auxiliary_losses,
                    auxiliary_weights=self.auxiliary_weights,
                    model=self.model,
                )

                loss = loss_dict["total"]

                # Detect non-finite losses
                if not torch.isfinite(loss):
                    # Save diagnostics and skip step to avoid corrupting optimizer
                    diag_dir = None
                    try:
                        save_root = None
                        if self.config.checkpoint and self.config.checkpoint.save_dir:
                            save_root = self.config.checkpoint.save_dir
                        if save_root:
                            diag_dir = Path(save_root) / "diag_nan_losses"
                            diag_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        diag_dir = None

                    try:
                        save_root = None
                        if self.config.checkpoint and self.config.checkpoint.save_dir:
                            save_root = self.config.checkpoint.save_dir
                        if save_root is None:
                            raise RuntimeError(
                                f"Non-finite loss detected at epoch {self.current_epoch}, step {self.global_step}, batch {batch_idx}. No checkpoint save_dir available to store diagnostics."
                            )
                        if diag_dir is None:
                            diag_dir = Path(save_root) / "diag_nan_losses"
                            diag_dir.mkdir(parents=True, exist_ok=True)

                        torch.save(
                            {
                                "epoch": int(self.current_epoch),
                                "step": int(self.global_step),
                                "batch_idx": int(batch_idx),
                                "features": batch.get("features"),
                                "labels": batch.get("labels"),
                                "loss_dict": {
                                    k: float(v.item()) if isinstance(
                                        v, torch.Tensor) else float(v)
                                    for k, v in loss_dict.items()
                                },
                            },
                            diag_dir /
                            f"bad_loss_epoch{self.current_epoch}_step{self.global_step}_batch{batch_idx}.pt",
                        )
                    except Exception as e:
                        self.accelerator.print(
                            f"Failed saving loss diagnostics: {e}")
                    self.accelerator.print(
                        f"⚠️ Non-finite loss detected at epoch {self.current_epoch}, step {self.global_step}, batch {batch_idx}. Raising RuntimeError to surface the issue."
                    )
                    # If configured to skip extremely high losses we may also offer
                    # an option to skip non-finite losses; default behavior is to
                    # raise so failures surface during debugging.
                    skip_non_finite = False
                    try:
                        safeguards = getattr(self.config, "safeguards", {}) or {}
                        skip_non_finite = bool(safeguards.get("skip_non_finite_losses", False))
                    except Exception:
                        skip_non_finite = False

                    if skip_non_finite:
                        self.accelerator.print(
                            f"⚠️ Non-finite loss detected at epoch {self.current_epoch}, step {self.global_step}, batch {batch_idx}. Skipping step per safeguards configuration."
                        )
                        # Best-effort save (already attempted above) and continue
                        try:
                            # Zero gradients to avoid optimizer contamination
                            try:
                                self.optimizer.zero_grad(set_to_none=True)
                            except Exception:
                                self.optimizer.zero_grad()
                        except Exception:
                            pass
                        # Call batch end callback and increment global_step
                        self.callback_handler.on_batch_end(self, batch, batch_idx, None)
                        self.global_step += 1
                        continue

                    # Default behavior (do not silently continue)
                    raise RuntimeError(
                        f"Non-finite loss detected at epoch {self.current_epoch}, step {self.global_step}, batch {batch_idx}. See diag files if saved."
                    )
                    
                try:
                    safeguards = getattr(self.config, "safeguards", {}) or {}
                    max_loss = float(safeguards.get("max_loss", 1e6))
                except Exception:
                    max_loss = 1e6

                try:
                    loss_val = float(loss.item())
                except Exception:
                    loss_val = None

                if loss_val is not None and loss_val > max_loss:
                    # Save diagnostics for large loss (do NOT skip the step)
                    self.accelerator.print(
                        f"⚠️ Extremely large loss {loss_val:.3e} > max_loss {max_loss:.3e} detected; saving diagnostics and continuing optimizer update for epoch {self.current_epoch}, step {self.global_step}, batch {batch_idx}."
                    )
                    try:
                        save_root = None
                        if self.config.checkpoint and self.config.checkpoint.save_dir:
                            save_root = self.config.checkpoint.save_dir
                        if save_root:
                            diag_dir = Path(save_root) / "diag_high_losses"
                            diag_dir.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                {
                                    "epoch": int(self.current_epoch),
                                    "step": int(self.global_step),
                                    "batch_idx": int(batch_idx),
                                    "features": batch.get("features"),
                                    "labels": batch.get("labels"),
                                    "loss_dict": {
                                        k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                                        for k, v in loss_dict.items()
                                    },
                                },
                                diag_dir / f"high_loss_epoch{self.current_epoch}_step{self.global_step}_batch{batch_idx}.pt",
                            )
                    except Exception as e:
                        self.accelerator.print(f"Failed saving high-loss diagnostics: {e}")

                    # Decide whether to skip this step. By default we enable
                    # distributed-safe skipping to avoid a single rank corrupting
                    # model state when it observes an extremely large loss.
                    safeguards = getattr(self.config, "safeguards", {}) or {}
                    # Backwards-compatible default: enable ddp-safe skipping for
                    # very large losses unless explicitly disabled.
                    skip_on_high_loss = bool(safeguards.get("skip_on_high_loss", True))
                    ddp_sync_skip = bool(safeguards.get("ddp_sync_skip", True))

                    # Local decision: do we want to skip because of a large loss?
                    local_skip = skip_on_high_loss

                    should_skip_global = local_skip
                    if ddp_sync_skip and getattr(self.accelerator, "num_processes", 1) > 1:
                        try:
                            # Use accelerator.reduce to compute the logical OR
                            # across processes (max reduction of integer flag).
                            skip_tensor = torch.tensor(
                                1 if local_skip else 0,
                                dtype=torch.int,
                                device=self.accelerator.device,
                            )
                            reduced = self.accelerator.reduce(skip_tensor, reduction="max")
                            if isinstance(reduced, torch.Tensor):
                                should_skip_global = bool(int(reduced.item()))
                            else:
                                try:
                                    # Try converting to tensor and read scalar
                                    should_skip_global = bool(
                                        int(torch.as_tensor(
                                            reduced, device=self.accelerator.device
                                        ).item())
                                    )
                                except Exception:
                                    # If conversion fails, fall back to local decision
                                    should_skip_global = local_skip
                        except Exception:
                            # Fall back to local decision if reduce is unavailable
                            should_skip_global = local_skip

                    if should_skip_global:
                        # Save an explicit message and ensure optimizer state isn't corrupted.
                        self.accelerator.print(
                            "⚠️ High loss detected on at least one process — skipping optimizer update for this step across all processes (DDP-safe skip)."
                        )

                        # Make a best-effort to save per-process diagnostics if not already
                        # produced above; this is best-effort and should not cause the
                        # training to fail.
                        try:
                            save_root = None
                            if self.config.checkpoint and self.config.checkpoint.save_dir:
                                save_root = self.config.checkpoint.save_dir
                            if save_root:
                                diag_dir = Path(save_root) / "diag_skipped_steps"
                                diag_dir.mkdir(parents=True, exist_ok=True)
                                torch.save(
                                    {
                                        "epoch": int(self.current_epoch),
                                        "step": int(self.global_step),
                                        "batch_idx": int(batch_idx),
                                        "loss": loss_val,
                                        "features": batch.get("features"),
                                        "labels": batch.get("labels"),
                                    },
                                    diag_dir / f"skip_epoch{self.current_epoch}_step{self.global_step}_batch{batch_idx}.pt",
                                )
                        except Exception as e:
                            self.accelerator.print(f"Failed saving skip diagnostics: {e}")

                        # Clear accumulated gradients to avoid contaminating future steps
                        try:
                            self.optimizer.zero_grad(set_to_none=True)
                        except Exception:
                            self.optimizer.zero_grad()

                        # Call batch-end callbacks with outputs so callbacks can record
                        # the skip for tracing / debugging. Increase global_step so
                        # bookkeeping remains consistent.
                        self.callback_handler.on_batch_end(self, batch, batch_idx, outputs)
                        self.global_step += 1

                        # Continue to next batch — do not call optimizer.step() on any rank
                        continue

                # Backward pass
                self.accelerator.backward(loss)

                self.callback_handler.on_backward_end(self)

                invalid_grad = False
                grad_norm = None
                param_norm = None
                try:
                    total_norm_sq = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            if not torch.isfinite(p.grad).all():
                                invalid_grad = True
                                break
                            # accumulate gradient norm squared (use float to avoid overflow)
                            g = p.grad.detach()
                            total_norm_sq += float(g.float().norm() ** 2)

                    # Compute parameter norms for diagnostics
                    param_total_norm_sq = 0.0
                    for p in self.model.parameters():
                        param_total_norm_sq += float(
                            p.detach().float().norm() ** 2)
                    param_norm = math.sqrt(param_total_norm_sq)

                    grad_norm = math.sqrt(total_norm_sq)
                except Exception:
                    invalid_grad = True

                if invalid_grad:
                    # Save diagnostics and raise so the failure surfaces
                    self.accelerator.print(
                        f"⚠️ Non-finite gradients detected at epoch {self.current_epoch}, step {self.global_step}. Saving diagnostics and raising RuntimeError."
                    )
                    try:
                        save_root = None
                        if self.config.checkpoint and self.config.checkpoint.save_dir:
                            save_root = self.config.checkpoint.save_dir
                        if save_root is None:
                            raise RuntimeError(
                                f"Non-finite gradients detected at epoch {self.current_epoch}, step {self.global_step}. No checkpoint save_dir available to store diagnostics."
                            )
                        diag_dir = Path(save_root) / "diag_nan_grads"
                        diag_dir.mkdir(parents=True, exist_ok=True)
                        # Save a subset of gradients (first 10 param grads) to avoid huge files
                        grad_dump = {}
                        for i, p in enumerate(self.model.parameters()):
                            if p.grad is not None and i < 10:
                                grad_dump[f"grad_{i}"] = p.grad.detach().cpu()
                        torch.save(
                            {
                                "epoch": int(self.current_epoch),
                                "step": int(self.global_step),
                                "batch_idx": int(batch_idx),
                                "grads": grad_dump,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                "optimizer_state": {
                                    k: v for k, v in self.optimizer.state_dict().items() if k in ["state", "param_groups"]
                                },
                            },
                            diag_dir /
                            f"bad_grads_epoch{self.current_epoch}_step{self.global_step}_batch{batch_idx}.pt",
                        )
                    except Exception as e:
                        self.accelerator.print(
                            f"Failed saving gradient diagnostics: {e}")
                    raise RuntimeError(
                        f"Non-finite gradients detected at epoch {self.current_epoch}, step {self.global_step}. See diag files if saved."
                    )

                # If gradient norm is extraordinarily large, log it and skip the update
                if grad_norm is not None and grad_norm > 1e5:
                    # Save diagnostics and raise to capture where gradient explosion came from
                    self.accelerator.print(
                        f"⚠️ Extremely large gradient norm {grad_norm:.3e} detected; raising RuntimeError to surface the issue."
                    )
                    try:
                        save_root = None
                        if self.config.checkpoint and self.config.checkpoint.save_dir:
                            save_root = self.config.checkpoint.save_dir
                        if save_root is None:
                            raise RuntimeError(
                                f"Extremely large gradient norm {grad_norm:.3e} detected at epoch {self.current_epoch}, step {self.global_step}, but no checkpoint save_dir available to store diagnostics."
                            )
                        diag_dir = Path(save_root) / "diag_grad_norms"
                        diag_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "epoch": int(self.current_epoch),
                                "step": int(self.global_step),
                                "batch_idx": int(batch_idx),
                                "grad_norm": grad_norm,
                                "param_norm": param_norm,
                                "lr": self.optimizer.param_groups[0]["lr"],
                            },
                            diag_dir /
                            f"bad_grad_norm_epoch{self.current_epoch}_step{self.global_step}_batch{batch_idx}.pt",
                        )
                    except Exception as e:
                        self.accelerator.print(
                            f"Failed saving grad_norm diagnostics: {e}")
                    raise RuntimeError(
                        f"Extremely large gradient norm {grad_norm:.3e} detected at epoch {self.current_epoch}, step {self.global_step}. See diag files if saved."
                    )

                # Gradient clipping (if not using callback). Be explicit about
                # types: a boolean value in the config (eg. `gradient_clipping: true`)
                # should not be treated as a numeric clip value (True == 1.0),
                # because that silently enforces clipnorm=1.0. Instead warn and
                # skip clipping unless a numeric max_norm is provided.
                if (
                    self.config.gradient_clipping is not None
                    and not any(hasattr(cb, "max_norm") for cb in self.callback_handler.callbacks)
                ):
                    gc_val = self.config.gradient_clipping
                    if isinstance(gc_val, bool):
                        # Avoid treating boolean True as 1.0 which leads to
                        # surprising behavior. Advise the user to provide a
                        # numeric value if clipping is desired.
                        self.accelerator.print(
                            "⚠️ `gradient_clipping` set to a boolean in config — expected a numeric max_norm (e.g. 0.5). Skipping clipping."
                        )
                    else:
                        try:
                            max_norm = float(gc_val)
                            # Prefer accelerator's clip helper (handles distributed models)
                            try:
                                self.accelerator.clip_grad_norm_(
                                    self.model.parameters(), max_norm
                                )
                            except Exception:
                                # Fallback to native PyTorch API
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm
                                )
                        except Exception:
                            self.accelerator.print(
                                f"⚠️ Invalid gradient_clipping value: {gc_val}. Skipping clipping."
                            )

                # Optimizer step
                self.optimizer.step()

                # --- Parameter diagnostics: check for NaN/Inf in model params after update
                params_invalid = False
                try:
                    for p in self.model.parameters():
                        if not torch.isfinite(p).all():
                            params_invalid = True
                            break
                except Exception:
                    params_invalid = True

                if params_invalid:
                    self.accelerator.print(
                        f"⚠️ Non-finite model parameter detected after optimizer.step() at epoch {self.current_epoch}, step {self.global_step}. Aborting training."
                    )
                    # Save model state & batch for debugging
                    try:
                        save_root = None
                        if self.config.checkpoint and self.config.checkpoint.save_dir:
                            save_root = self.config.checkpoint.save_dir
                        if save_root:
                            diagnostic_path = Path(
                                save_root) / f"param_anomaly_epoch{self.current_epoch}_step{self.global_step}.pt"
                            torch.save({
                                "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                                "batch": {"features": batch.get("features"), "labels": batch.get("labels")},
                            }, diagnostic_path)
                            self.accelerator.print(
                                f"Saved parameter diagnostics to: {diagnostic_path}")
                    except Exception:
                        pass
                    # Stop training to avoid further corruption
                    raise RuntimeError(
                        "Detected NaN/Inf in model parameters after update. See saved diagnostics.")

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

                # Add grad/param norms if available
                try:
                    if grad_norm is not None:
                        metrics["grad_norm"] = float(grad_norm)
                    if param_norm is not None:
                        metrics["param_norm"] = float(param_norm)
                except Exception:
                    pass

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
                self.save_checkpoint()

            # Check if we've reached max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        avg_loss = total_loss / num_batches

        # Sanity check for iterable datasets: warn if the dataloader yielded fewer
        # batches than expected. This is often due to IterableDataset / sharding
        # semantics (unequal partitions), rounding by batch sizes, or missing
        # examples in the manifest.
        if num_batches != steps_per_gpu:
            # Only print from local main process to avoid spamming logs
            if self.accelerator.is_local_main_process:
                self.accelerator.print(
                    f"⚠️  Observed {num_batches} batches on this process, expected {steps_per_gpu} per process. "
                    "This may be normal for IterableDataset sharding, but verify dataset size and partitioning if unexpected."
                )
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

            num_val_batches = 0
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

                outputs = self.model(
                    x=batch["features"],
                    is_target=batch["is_target"],
                    labels=batch["labels"],
                )

                # Diagnostic checks for validation as well
                try:
                    logits = outputs if isinstance(
                        outputs, torch.Tensor) else outputs.logits
                    if not torch.isfinite(logits).all():
                        self.accelerator.print(
                            f"⚠️ Non-finite logits detected during validation at step {batch_idx} for split {split_name}. Skipping batch."
                        )
                        continue
                except Exception:
                    pass

                if "labels" in batch:
                    targets = batch["labels"]
                    loss = self.loss_fn(outputs[:, 1:, :], targets)
                    total_samples += targets.size(0)
                elif "speaker_activity" in batch:
                    targets = batch["speaker_activity"]
                    loss = self.loss_fn(outputs[:, 1:, :], targets)
                    total_samples += targets.size(0)
                else:
                    loss = None
                    targets = None

                if loss is not None:
                    total_loss += loss.item()

                # Compute metrics, ignoring the enrollment token
                if targets is not None:
                    batch_metrics = compute_metrics(
                        outputs=outputs[:, 1:, :],
                        targets=targets,
                        task_type="classification",
                    )
                    all_metrics.append(batch_metrics)
                num_val_batches += 1

            if num_val_batches == 0:
                split_results = {"val_loss": float("nan")}
            else:
                split_results = {"val_loss": total_loss / num_val_batches}

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

    def save_checkpoint(self):
        """Save training checkpoint using Accelerate."""
        if not self.config.checkpoint or not self.config.checkpoint.save_dir:
            return

        self.accelerator.save_state()
        self.accelerator.print(f"✅ Checkpoint saved by Accelerate.")

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

            # Check for non-finite logits in test run
            try:
                logits = outputs if isinstance(
                    outputs, torch.Tensor) else outputs.logits
                if not torch.isfinite(logits).all():
                    self.accelerator.print(
                        "⚠️ Non-finite logits detected during testing. Skipping batch."
                    )
                    continue
            except Exception:
                pass

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
                task_type="classification",
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
