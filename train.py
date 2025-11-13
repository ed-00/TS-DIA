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
from utility.dataset_utils import prepare_training_cuts


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
    if training_config.training_dataset_map:
        train_cuts = prepare_training_cuts(
            cut_sets, training_config.training_dataset_map)
    else:
        # Fallback to old behavior if training_dataset_map is not provided
        dataset_name = dataset_configs[0].name
        dataset_cuts = cut_sets[dataset_name]
        # TODO: make this more robust
        train_cuts = dataset_cuts.get("train_b2_mix100000")

    if train_cuts is None:
        raise ValueError(f"No training data found for dataset.")

    # Get label type from training config
    label_type = training_config.eval_knobs.get("label_type", "binary")

    # Get random seed (training config takes precedence over global config)
    random_seed = training_config.random_seed or global_config.random_seed

    # Prepare validation datasets
    val_dataloaders: Dict[str, DataLoader] = {}

    if training_config.validation:
        if training_config.validation.validation_dataset_map:
            # Use the new validation_dataset_map approach
            if training_config.validation.validation_dataset_map.combine:
                # Combine all validation splits into one
                val_cuts = prepare_training_cuts(
                    cut_sets, training_config.validation.validation_dataset_map)
                val_dataloaders["val"] = DatasetManager.create_dataloader(
                    cuts=val_cuts,
                    data_loading=global_config.data_loading,
                    batch_size=training_config.batch_size,
                    label_type=label_type,
                    random_seed=random_seed,
                    shuffle=False,
                )
            else:
                # Keep datasets separate
                for split_info in training_config.validation.validation_dataset_map.splits:
                    dataset_name = split_info.dataset_name
                    split_name = split_info.split_name
                    subset_ratio = split_info.subset_ratio

                    if dataset_name not in cut_sets:
                        print(
                            f"Warning: Dataset '{dataset_name}' not found for validation, skipping.")
                        continue

                    dataset_cuts = cut_sets[dataset_name]
                    if split_name not in dataset_cuts:
                        print(
                            f"Warning: Split '{split_name}' not found in dataset '{dataset_name}', skipping.")
                        continue

                    val_cuts = dataset_cuts[split_name]

                    # Apply subsetting if needed
                    if 0 < subset_ratio < 1.0:
                        num_cuts = int(len(val_cuts) * subset_ratio)
                        val_cuts = val_cuts.subset(first=num_cuts)

                    # Create separate dataloader for each validation split
                    val_key = f"{dataset_name}_{split_name}"
                    val_dataloaders[val_key] = DatasetManager.create_dataloader(
                        cuts=val_cuts,
                        data_loading=global_config.data_loading,
                        batch_size=training_config.batch_size,
                        label_type=label_type,
                        random_seed=random_seed,
                        shuffle=False,
                    )
        else:
            # Fallback to old behavior using splits list
            requested_splits = training_config.validation.splits or []
            dataset_name = dataset_configs[0].name
            dataset_cuts = cut_sets.get(dataset_name, {})

            for split in requested_splits:
                if not split:
                    continue
                split_cuts = dataset_cuts.get(split)
                if split_cuts is None:
                    print(
                        f"Warning: Requested validation split '{split}' not found, skipping.")
                    continue

                val_dataloaders[split] = DatasetManager.create_dataloader(
                    cuts=split_cuts,
                    data_loading=global_config.data_loading,
                    batch_size=training_config.batch_size,
                    label_type=label_type,
                    random_seed=random_seed,
                    shuffle=False,
                )

    # Create diarization dataloaders using DatasetManager
    train_dataloader = DatasetManager.create_dataloader(
        cuts=train_cuts,
        data_loading=global_config.data_loading,
        batch_size=training_config.batch_size,
        label_type=label_type,
        random_seed=random_seed,
        shuffle=True,
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
