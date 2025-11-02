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
from data_manager.data_manager import DatasetManager
from model.model_factory import create_model
from parse_args import unified_parser
from training import Trainer


def main():
    """Main training function."""
    # Parse unified configuration with CLI overrides
    args, model_config, dataset_configs, training_config, config_path = unified_parser()

    # Load datasets and create diarization dataloaders (parser ensures valid configs)
    cut_sets = DatasetManager.load_datasets(datasets=dataset_configs)

    # If there's no training configuration provided, stop after data prep
    if training_config is None:
        print(
            "No training configuration found — datasets downloaded and prepared. Exiting.")
        return

    # Get dataset name and configuration
    dataset_name = dataset_configs[0].name
    global_config = dataset_configs[0].global_config

    # Get unified train/val splits (DatasetManager handles split mapping)
    dataset_cuts = cut_sets[dataset_name]
    train_cuts = dataset_cuts.get("train")
    val_cuts = dataset_cuts.get("val")

    # Get label type from training config
    label_type = training_config.eval_knobs.get("label_type", "binary")

    # Get random seed (training config takes precedence over global config)
    random_seed = training_config.random_seed or global_config.random_seed

    # Create diarization dataloaders using DatasetManager
    train_dataloader, val_dataloader = DatasetManager.create_train_val_dataloaders(
        train_cuts=train_cuts,
        val_cuts=val_cuts,
        data_loading=global_config.data_loading,
        label_type=label_type,
        random_seed=random_seed,
    )
    print(f"\n✓ Diarization dataloaders created with label_type='{label_type}'")

    # Create model (parser ensures configs are never None)
    model = create_model(model_config)
    print(f"Model created: {model_config.name}")
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=training_config,
        config_path=config_path,
    )

    # Start training
    trainer.train()

    # Run final test if available
    if trainer.test_dataloader:
        trainer.test()

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
