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
from data_manager.data_manager import DatasetManager
from model.model_factory import create_model
from parse_args import unified_parser
from training import Trainer
from training.accelerate_utils import setup_accelerator, print_training_info
from training.logging_utils import log_system_info, setup_file_logger


def main():
    """Main training function."""
    # Parse unified configuration with CLI overrides
    _, model_config, dataset_configs, global_config, training_config, config_path = unified_parser()

    if dataset_configs is None or len(dataset_configs) == 0 or not global_config:
        raise ValueError("No data config parsed")

    # Initialize Accelerator FIRST (before any other operations)
    if training_config is None:
        raise ValueError("Training configuration is required")

    if training_config.training_dataset_map is None:
        raise ValueError(
            "Training dataset mapping is required in training configuration")

    accelerator = setup_accelerator(
        training_config,
        project_dir=training_config.checkpoint.save_dir if training_config.checkpoint else None,
    )

    # Log system info and training configuration
    log_system_info(accelerator)
    print_training_info(accelerator, training_config)

    # Create model (parser ensures configs are never None)
    if model_config is None:
        raise ValueError("No model configuration found")
    accelerator.print("\nCreating model...")
    model = create_model(model_config)
    accelerator.print(f"✓ Model created: {model_config.name}")

    train_dataloader = None
    val_dataloaders = None
    DataManager = DatasetManager()
    cut_sets = None
   
    # Load datasets and create diarization dataloaders (parser ensures valid configs)
    accelerator.print("\n" + "="*70)
    accelerator.print("Loading Datasets")
    accelerator.print("="*70)
    cut_sets = DataManager.load_datasets(
        datasets=dataset_configs,
        global_config=global_config,
        cache_dir=global_config.cache_dir if global_config.cache_dir else None,
        training_dataset_mapping=training_config.training_dataset_map
    )
    accelerator.print(f"✓ Loaded {len(cut_sets)} dataset(s)")

    if global_config is None:
        raise ValueError("No global configuration found in dataset config")

    # Validate training dataset map
    if training_config.training_dataset_map is None:
        raise ValueError("No training dataset map was found for dataset.")

    # Get label type from training config
    label_type = training_config.eval_knobs.get("label_type", "ego")

    # Get random seed (training config takes precedence over global config)
    random_seed = training_config.random_seed or global_config.random_seed

    accelerator.print("\nCreating dataloaders...")
    train_dataloader = DataManager.create_training_dataloader(
        cut_sets=cut_sets,
        global_config=global_config,
        training_config=training_config,
        label_type=label_type,
        random_seed=random_seed
    )

    # Prepare validation datasets using DatasetManager
    val_dataloaders = DataManager.create_validation_dataloaders(
        cut_sets=cut_sets,
        dataset_configs=dataset_configs,
        global_config=global_config,
        training_config=training_config,
        label_type=label_type,
        random_seed=random_seed
    )
    accelerator.print(
        f"\n✓ Diarization dataloaders created with label_type='{label_type}'")
    if val_dataloaders:
        accelerator.print(
            f"  Validation splits: {', '.join(val_dataloaders.keys())}")
    # Create trainer (pass accelerator to prevent re-initialization)
    accelerator.print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloaders if val_dataloaders else None,
        config=training_config,
        config_path=str(config_path),
        accelerator=accelerator
    )

    # Start training
    accelerator.print("\n" + "="*70)
    accelerator.print("Starting Training")
    accelerator.print("="*70 + "\n")
    trainer.train()

    # Run final test if available
    if trainer.test_dataloader:
        accelerator.print("\nRunning final test...")
        trainer.test()

    accelerator.print("\n" + "="*70)
    accelerator.print("Training completed successfully!")
    accelerator.print("="*70)


if __name__ == "__main__":
    main()
