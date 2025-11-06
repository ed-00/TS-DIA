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
from typing import Dict, List

from torch.utils.data import DataLoader

from data_manager.data_manager import DatasetManager
from model.model_factory import create_model
from parse_args import unified_parser
from training import Trainer
from data_manager.dataset_types import DatasetConfig


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

    # Get dataset name and configuration
    dataset_name: str = dataset_configs[0].name

    if global_config is None:
        raise ValueError("No global configuration found in dataset config")

    # Get unified train/val splits (DatasetManager handles split mapping)
    dataset_cuts = cut_sets[dataset_name]
    train_cuts = dataset_cuts.get("train")

    if train_cuts is None:
        raise ValueError(f"No training data found for dataset {dataset_name}")

    # Determine requested validation splits (defaults to ['val'] when available)
    validation_splits: List[str] = []
    if training_config.validation:
        requested_splits = training_config.validation.splits or []
        for split in requested_splits:
            if split and split not in validation_splits:
                validation_splits.append(split)
    elif dataset_cuts.get("val") is not None:
        validation_splits = ["val"]

    # Get label type from training config
    label_type = training_config.eval_knobs.get("label_type", "binary")

    # Get random seed (training config takes precedence over global config)
    random_seed = training_config.random_seed or global_config.random_seed

    # Create diarization dataloaders using DatasetManager
    train_dataloader = DatasetManager.create_dataloader(
        cuts=train_cuts,
        data_loading=global_config.data_loading,
        batch_size=training_config.batch_size,
        label_type=label_type,
        random_seed=random_seed,
        shuffle=True,
    )

    val_dataloaders: Dict[str, DataLoader] = {}
    missing_splits: List[str] = []

    for split in validation_splits:
        split_cuts = dataset_cuts.get(split)
        if split_cuts is None:
            missing_splits.append(split)
            continue

        val_dataloaders[split] = DatasetManager.create_dataloader(
            cuts=split_cuts,
            data_loading=global_config.data_loading,
            batch_size=training_config.batch_size,
            label_type=label_type,
            random_seed=random_seed,
            shuffle=False,
        )

    if missing_splits:
        if validation_splits == ["val"] and missing_splits == ["val"]:
            val_dataloaders.clear()
            print("Warning: Requested validation split 'val' not found; skipping validation.")
        else:
            available = ", ".join(sorted(dataset_cuts.keys()))
            raise ValueError(
                f"Requested validation split(s) {missing_splits} not found for dataset {dataset_name}. "
                f"Available splits: {available}"
            )

    print(f"\n✓ Diarization dataloaders created with label_type='{label_type}'")
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
