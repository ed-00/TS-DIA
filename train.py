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
import os

from data_manager.data_manager import DatasetManager
from model.model_factory import create_model
from parse_args import unified_parser
from training import Trainer
from training.diarization_dataloader import create_train_val_dataloaders


def main():
    """Main training function."""
    # Parse unified configuration with CLI overrides
    args, model_config, dataset_configs, training_config, config_path = unified_parser()

    # Load datasets and create diarization dataloaders (parser ensures valid configs)
    cut_sets = DatasetManager.load_datasets(datasets=dataset_configs)

    # If there's no training configuration provided, stop after data prep
    if training_config is None:
        print("No training configuration found — datasets downloaded and prepared. Exiting.")
        return

    # Extract feature configuration (parser guarantees it's never None)
    feature_config = dataset_configs[0].global_config.get_feature_config()

    dataset_name = dataset_configs[0].name
    base_storage_path = dataset_configs[0].global_config.storage_path

    print(
        f"Using feature config: {feature_config.feature_type} with {feature_config.num_mel_bins} bins"
    )

    # Get unified train/val splits (DatasetManager handles split mapping)
    dataset_cuts = cut_sets[dataset_name]
    train_cuts = dataset_cuts.get("train")
    val_cuts = dataset_cuts.get("val")

    # Create diarization dataloaders
    label_type = training_config.eval_knobs.get("label_type", "binary")
    max_duration = training_config.eval_knobs.get("max_duration", None)
    drop_last = (
        training_config.performance.drop_last if training_config.performance else False
    )
    min_speaker_dim = training_config.eval_knobs.get("min_speaker_dim", None)

    train_dataloader, val_dataloader, train_cuts_with_feats, val_cuts_with_feats = (
        create_train_val_dataloaders(
            train_cuts=train_cuts,
            val_cuts=val_cuts,
            batch_size=training_config.batch_size,
            val_batch_size=training_config.validation.batch_size
            if training_config.validation
            else training_config.batch_size,
            num_workers=training_config.performance.num_workers
            if training_config.performance
            else 0,
            max_duration=max_duration,
            label_type=label_type,
            pin_memory=training_config.performance.pin_memory
            if training_config.performance
            else True,
            feature_config=feature_config,
            dataset_name=dataset_name,
            base_storage_path=base_storage_path,
            drop_last=drop_last,
            min_speaker_dim=min_speaker_dim,
            data_loading=dataset_configs[0].global_config.data_loading,
            random_seed=training_config.random_seed
            or dataset_configs[0].global_config.random_seed,
        )
    )
    print(f"Diarization dataloaders created with label_type='{label_type}'")
    if training_config is None:
        print("No training configuration provided. Preprocessing complete.")
        return
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
