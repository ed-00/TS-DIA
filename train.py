#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Universal Training Script

This script provides a complete training pipeline for deep learning models
with support for distributed training, mixed precision, callbacks, and more.

Usage:
    # Basic training with config file
    python train.py --config configs/training_example.yml

    # With CLI overrides
    python train.py --config configs/training_example.yml \\
        --epochs 100 \\
        --batch-size 64 \\
        --lr 1e-4

    # Distributed training with Accelerate
    accelerate launch train.py --config configs/training_example.yml

    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=4 train.py --config configs/training_example.yml
"""
import sys
from pathlib import Path
from typing import Dict, List

from torch.utils.data import DataLoader

from data_manager.data_manager import DatasetManager
from model.model_factory import create_model
from parse_args import unified_parser
from training import Trainer


def main():
    """Main training function."""
    # Parse unified configuration with CLI overrides
    _, model_config, dataset_configs, global_config, training_config, config_path = unified_parser()

    if dataset_configs is None or len(dataset_configs) == 0 or not global_config:
        raise ValueError("No data config parsed")

    # Load datasets and create diarization dataloaders (parser ensures valid configs)
    cut_sets = DatasetManager.load_datasets(datasets=dataset_configs)

    # If there's no training configuration provided, stop after data prep
    if training_config is None:
        print(
            "No training configuration found — datasets downloaded and prepared. Exiting.")
        sys.exit(0)

    if global_config is None:
        raise ValueError("No global configuration found in dataset config")

    # Validate training dataset map
    if training_config.training_dataset_map is None:
        raise ValueError("No training dataset map was found for dataset.")

    # Get label type from training config
    label_type = training_config.eval_knobs.get("label_type", "ego")

    # Get random seed (training config takes precedence over global config)
    random_seed = training_config.random_seed or global_config.random_seed

    # Create training dataloader using DatasetManager
    train_dataloader = DatasetManager.create_training_dataloader(
        cut_sets=cut_sets,
        global_config=global_config,
        training_config=training_config,
        label_type=label_type,
        random_seed=random_seed,
    )

    # Prepare validation datasets using DatasetManager
    val_dataloaders = DatasetManager.create_validation_dataloaders(
        cut_sets=cut_sets,
        dataset_configs=dataset_configs,
        global_config=global_config,
        training_config=training_config,
        label_type=label_type,
        random_seed=random_seed,
    )

    print(
        f"\n✓ Diarization dataloaders created with label_type='{label_type}'")
    if val_dataloaders:
        print(f"  Validation splits: {', '.join(val_dataloaders.keys())}")

    # Create model (parser ensures configs are never None)
    if model_config is None:
        raise ValueError("No model configuration found")
    model = create_model(model_config)
    print(f"Model created: {model_config.name}")
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloaders if val_dataloaders else None,
        config=training_config,
        config_path=str(config_path),
    )

    # Start training
    trainer.train()

    # Run final test if available
    if trainer.test_dataloader:
        trainer.test()

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
