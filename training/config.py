#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Training Configuration Dataclasses

This module defines typed dataclasses for all training configurations.
All settings from YAML are mapped to these dataclasses for type safety,
validation, and easy serialization.

Key Classes:
    OptimizerConfig: Optimizer settings (type, lr, weight_decay, etc.)
    SchedulerConfig: Learning rate scheduler settings
    LossConfig: Loss function configuration with auxiliary losses
    CheckpointConfig: Checkpoint saving/loading settings
    ValidationConfig: Validation interval and batch size
    EarlyStoppingConfig: Early stopping criteria
    LoggingConfig: Logging backend and metrics configuration
    PerformanceConfig: DataLoader performance settings
    DistributedConfig: Distributed training settings
    TrainingConfig: Complete training configuration
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OptimizerConfig:
    """
    Optimizer configuration.

    Attributes:
        type: Optimizer type (adam, adamw, sgd, adagrad, rmsprop, etc.)
        lr: Learning rate
        weight_decay: Weight decay / L2 regularization
        betas: Beta parameters for Adam-style optimizers
        momentum: Momentum for SGD
        epsilon: Epsilon for numerical stability
        amsgrad: Use AMSGrad variant for Adam
        nesterov: Use Nesterov momentum for SGD
    """

    type: str
    lr: float
    weight_decay: Optional[float] = 0.0
    betas: Optional[Tuple[float, float]] = (0.9, 0.999)
    momentum: Optional[float] = None
    epsilon: Optional[float] = 1e-8
    amsgrad: bool = False
    nesterov: bool = False


@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration.

    Attributes:
        type: Scheduler type (cosine, linear, exponential, step, plateau, etc.)
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate (for cyclic schedulers)
        warmup_steps: Number of warmup steps
        decay_steps: Number of decay steps
        num_cycles: Number of cycles (for cosine with restarts)
        step_size: Step size for StepLR
        gamma: Multiplicative factor for decay
        patience: Patience for ReduceLROnPlateau
        mode: Mode for ReduceLROnPlateau (min/max)
    """

    type: str
    min_lr: float = 0.0
    max_lr: Optional[float] = None
    warmup_steps: int = 0
    decay_steps: Optional[int] = None
    num_cycles: int = 1
    step_size: Optional[int] = None
    gamma: float = 0.1
    patience: Optional[int] = None
    mode: str = "min"


@dataclass
class EarlyStoppingConfig:
    """
    Early stopping configuration.

    Attributes:
        patience: Number of epochs to wait for improvement
        metric: Metric to monitor (loss, accuracy, etc.)
        min_delta: Minimum change to qualify as improvement
        mode: Direction of improvement (min/max)
        restore_best_weights: Restore model to best checkpoint
    """

    patience: int
    metric: str
    min_delta: float = 0.0
    mode: str = "min"
    restore_best_weights: bool = True


@dataclass
class LossConfig:
    """
    Loss function configuration.

    Attributes:
        main: Main loss function (cross_entropy, mse, bce, etc.)
        label_smoothing: Label smoothing factor
        reduction: Loss reduction method (mean, sum, none)
        auxiliary: Dictionary of auxiliary losses with weights
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
    """

    main: str
    label_smoothing: Optional[float] = None
    reduction: str = "mean"
    auxiliary: Dict[str, float] = field(default_factory=dict)
    focal_alpha: Optional[float] = None
    focal_gamma: Optional[float] = None


@dataclass
class CheckpointConfig:
    """
    Checkpoint configuration.

    Attributes:
        save_dir: Directory to save checkpoints
        interval: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep
        resume: Path to checkpoint to resume from
        snapshot_optimizer: Save optimizer state
        snapshot_scheduler: Save scheduler state
        snapshot_features: Save Performer feature matrices
        save_best_only: Only save best checkpoint
        monitor_metric: Metric to monitor for best checkpoint
    """

    save_dir: str
    interval: int
    save_total_limit: Optional[int] = None
    resume: Optional[str] = None
    snapshot_optimizer: bool = True
    snapshot_scheduler: bool = True
    snapshot_features: bool = True
    save_best_only: bool = False
    monitor_metric: Optional[str] = None


@dataclass
class TrainingDatasetSplit:
    """
    Configuration for a single dataset split in the training map.

    Attributes:
        dataset_name: Name of the dataset.
        split_name: Name of the split (e.g., 'train', 'train_si284').
        subset_ratio: Fraction of the dataset to use (0.0 to 1.0).
    """

    dataset_name: str
    split_name: str
    subset_ratio: float = 1.0


@dataclass
class TrainingDatasetMap:
    """
    Configuration for mapping and combining training datasets.

    Attributes:
        combine: Whether to combine multiple datasets into one.
        splits: List of dataset splits to use for training.
    """

    combine: bool = True
    splits: List[TrainingDatasetSplit] = field(default_factory=list)


@dataclass
class ValidationConfig:
    """
    Validation configuration.

    Attributes:
        interval: Validate every N steps
        batch_size: Batch size for validation
        max_steps: Maximum number of validation batches (None for all)
        metric_for_best_model: Metric to track for best model
        greater_is_better: Whether higher metric is better
        splits: Optional list of validation split names to evaluate (deprecated, use validation_dataset_map)
        validation_dataset_map: Configuration for validation dataset mapping
    """

    interval: int
    batch_size: int
    max_steps: Optional[int] = None
    metric_for_best_model: str = "val_loss"
    greater_is_better: bool = False
    splits: List[str] = field(default_factory=lambda: ["val"])
    validation_dataset_map: Optional[TrainingDatasetMap] = None



@dataclass
class LoggingConfig:
    """
    Logging configuration.

    Attributes:
        interval: Log every N steps
        tensorboard: Enable TensorBoard logging
        wandb: Enable Weights & Biases logging
        log_metrics: List of metrics to log
        wandb_project: WandB project name
        wandb_entity: WandB entity/team name
        log_model: Log model architecture
    """

    interval: int
    tensorboard: bool = False
    wandb: bool = False
    log_metrics: List[str] = field(default_factory=list)
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_model: bool = False



@dataclass
class PerformanceConfig:
    """
    Performance optimization configuration.

    Attributes:
        num_workers: Number of DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        batch_shim: Use batch shim for irregular batch sizes
        prefetch_factor: Number of batches to prefetch
        persistent_workers: Keep workers alive between epochs
        compile_model: Use torch.compile for optimization
    """

    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    batch_shim: bool = False
    prefetch_factor: Optional[int] = 2
    persistent_workers: bool = False
    compile_model: bool = False


@dataclass
class DistributedConfig:
    """
    Distributed training configuration.

    Attributes:
        backend: Distributed backend (nccl, gloo, mpi)
        world_size: Total number of processes
        local_rank: Local rank of current process
        sync_gradient_barrier: Synchronize gradients across processes
        find_unused_parameters: Find unused parameters in DDP
        gradient_as_bucket_view: Optimize DDP memory usage
        static_graph: Use static graph optimization in DDP
    """

    backend: str = "nccl"
    world_size: int = 1
    local_rank: int = 0
    sync_gradient_barrier: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False


@dataclass
class TrainingConfig:
    """
    Complete training configuration.

    Attributes:
        epochs: Number of training epochs
        batch_size: Training batch size
        optimizer: Optimizer configuration
        scheduler: Scheduler configuration
        training_dataset_map: Configuration for training dataset mapping.
        gradient_clipping: Gradient clipping value (None to disable)
        gradient_accumulation_steps: Number of gradient accumulation steps
        mixed_precision: Enable mixed precision training
        amp_loss_scale: Loss scaling for AMP (None for dynamic)
        feature_redraw_interval: Redraw Performer features every N steps
        random_seed: Random seed for reproducibility
        fixed_projection: Use fixed projection for Performer
        early_stopping: Early stopping configuration
        validation: Validation configuration
        checkpoint: Checkpoint configuration
        loss: Loss function configuration
        distributed: Distributed training configuration
        logging: Logging configuration
        performance: Performance optimization configuration
        callbacks: List of callback names to use
        profiling: Enable PyTorch profiler
        eval_knobs: Evaluation-specific settings
        tuning: Hyperparameter tuning configuration
        max_steps: Maximum training steps (overrides epochs)
    """

    epochs: int
    batch_size: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training_dataset_map: Optional[TrainingDatasetMap] = None
    gradient_clipping: Optional[float] = None
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    amp_loss_scale: Optional[int] = None
    feature_redraw_interval: Optional[int] = None
    random_seed: int = 42
    fixed_projection: bool = False
    early_stopping: Optional[EarlyStoppingConfig] = None
    validation: Optional[ValidationConfig] = None
    checkpoint: Optional[CheckpointConfig] = None
    loss: Optional[LossConfig] = None
    distributed: Optional[DistributedConfig] = None
    logging: Optional[LoggingConfig] = None
    performance: Optional[PerformanceConfig] = None
    callbacks: List[str] = field(default_factory=list)
    profiling: bool = False
    eval_knobs: Dict[str, Any] = field(default_factory=dict)
    tuning: Dict[str, Any] = field(default_factory=dict)
    max_steps: Optional[int] = None
