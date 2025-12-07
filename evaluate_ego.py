#! /usr/bin/env python
from training.config import TrainingDatasetMap, TrainingDatasetSplit, ValidationConfig
from training.accelerate_utils import setup_accelerator
from training.trainer import Trainer
from training.config import TrainingConfig
from parse_args import parse_config
from model.model_factory import create_model
from data_manager.data_manager import DatasetManager
import os
import sys
import glob
import yaml
import torch
import pandas as pd
import wandb
import argparse
import re
from pathlib import Path
from tqdm.auto import tqdm
from safetensors.torch import load_file
from utility.collect_model import collect_model

# Add workspace root to path
sys.path.append(os.getcwd())


# Constants
MODEL_CONFIGS = [
    "configs/REPRODUCE_eval_only/02_pretraining_softmax.yaml",
    "configs/REPRODUCE_eval_only/03_pretraining_linear.yaml",
    "configs/REPRODUCE_eval_only/04_pretraining_linear_correct_nb-f.yaml"
]


from dacite import from_dict, Config


def load_training_config(config_path):
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if "training" in config_data:
        return from_dict(
            data_class=TrainingConfig, 
            data=config_data["training"],
            config=Config(cast=[tuple])
        )
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recursive_checkpoints", action="store_true", help="Recursively search for checkpoints in save_dir")
    args = parser.parse_args()

    # Initialize results list
    results = []

    # Initialize wandb
    wandb.init(project="ego_evaluation", name="eval_all_models_checkpoints")

    for config_path in MODEL_CONFIGS:
        print(f"\n{'='*80}")
        print(f"Processing Model Config: {config_path}")
        print(f"{'='*80}")

        # Parse Model and Training Config
        model_config, dataset_configs, train_global_config = parse_config(config_path)
        training_config = load_training_config(config_path)

        if not model_config or not training_config or not train_global_config or not dataset_configs:
            print(f"Skipping {config_path}: Missing configuration")
            continue

        accelerator = setup_accelerator(training_config)
        
        # Create Validation Mapping
        val_mapping = None
        if training_config.validation and training_config.validation.validation_dataset_map:
            val_mapping = training_config.validation.validation_dataset_map
        else:
            # Construct default validation mapping for simu datasets
            splits = []
            for ds in dataset_configs:
                if ds.name.startswith("simu"):
                    # Assuming standard dev split name for simu datasets
                    splits.append(TrainingDatasetSplit(dataset_name=ds.name, split_name="dev_b2_mix500"))
            
            if splits:
                val_mapping = TrainingDatasetMap(combine=False, splits=splits)
                print(f"Created default validation mapping for datasets: {[s.dataset_name for s in splits]}")
                
                # Inject into training config so create_validation_dataloaders uses it
                if not training_config.validation:
                    training_config.validation = ValidationConfig(interval=1, batch_size=1)
                training_config.validation.validation_dataset_map = val_mapping

        # Load Datasets
        print("Loading Datasets...")
        data_manager = DatasetManager()

        # We use the dataset configs from the model config
        # Pass empty training map and strict_splits=True to avoid loading training data
        cut_sets = data_manager.load_datasets(
            datasets=dataset_configs,
            global_config=train_global_config,
            training_dataset_mapping=TrainingDatasetMap(splits=[]), 
            validation_dataset_mapping=val_mapping,
            strict_splits=True
        )

        # Create Validation Dataloaders
        label_type = "ego"

        val_dataloaders = data_manager.create_validation_dataloaders(
            cut_sets=cut_sets,
            dataset_configs=dataset_configs,
            global_config=train_global_config,
            training_config=training_config,
            accelerator=accelerator,
            label_type=label_type
        )

        if not val_dataloaders:
            print("No validation dataloaders created. Check dataset config and splits.")
            continue
        if not training_config.checkpoint:
            raise ValueError("training_config.checkpoint: was not provided in the config...")
        
        # Find Checkpoints
        save_dir = training_config.checkpoint.save_dir
        if args.recursive_checkpoints:
            checkpoints = glob.glob(os.path.join(save_dir, "**", "checkpoint*"), recursive=True)
        else:
            checkpoints = glob.glob(os.path.join(save_dir, "checkpoint*"))

        
        # Sort by step number
        def get_step(path):
            name = os.path.basename(path)
            # Try to find the last number in the string
            match = re.search(r'(\d+)$', name)
            if match:
                return int(match.group(1))
            return 0

        checkpoints.sort(key=get_step)

        print(f"Found {len(checkpoints)} checkpoints in {save_dir} (recursive={args.recursive_checkpoints})")

        for ckpt_path in checkpoints:
            ckpt_name = os.path.basename(ckpt_path)
            print(f"Evaluating Checkpoint: {ckpt_name}")

            # Create Model
            model = create_model(model_config)

            # Load State Dict
            model_path = os.path.join(ckpt_path, "pytorch_model.bin")
            if not os.path.exists(model_path):
                model_path = os.path.join(ckpt_path, "model.safetensors")

            if os.path.exists(model_path):
                model = collect_model(model_path, model, accelerator.device)
            else:
                print(
                    f"Warning: Could not find model file in {ckpt_path}. Trying accelerator.load_state...")
                model = accelerator.prepare(model)
                try:
                    accelerator.load_state(ckpt_path)
                except Exception as e:
                    print(f"Failed to load checkpoint {ckpt_path}: {e}")
                    continue

            # Ensure model is on device
            model.to(accelerator.device)
            model.eval()

            # Create a temporary Trainer just for testing
            trainer = Trainer(
                model=model,
                config=training_config,
                accelerator=accelerator,
                val_dataloader=val_dataloaders,
            )

            # Run validation
            metrics = trainer.validate()

            # Log results
            result_entry = {
                "model_config": config_path,
                "model_name": model_config.name,
                "checkpoint": ckpt_name,
                **metrics
            }
            results.append(result_entry)

            # Wandb logging
            wandb.log(result_entry)

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        output_csv = "evaluation_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()
