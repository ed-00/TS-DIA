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
from training.losses import compute_loss, compute_metrics, create_loss_function, create_auxiliary_losses
from training.accelerate_utils import all_reduce_metrics
import numpy as np
import numbers
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from dacite import from_dict, Config

# Add workspace root to path
sys.path.append(os.getcwd())


# Constants
MODEL_CONFIGS = [
    "configs/REPRODUCE_eval_only/02_pretraining_softmax.yaml",
    "configs/REPRODUCE_eval_only/03_pretraining_linear.yaml",
    "configs/REPRODUCE_eval_only/04_pretraining_linear_correct_nb-f.yaml"
]




# Sort by step number
def get_step(path):
    name = os.path.basename(path)
    # Try to find the last number in the string
    match = re.search(r'(\d+)$', name)
    if match:
        return int(match.group(1))
    return 0
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


def validate_model(
    model: torch.nn.Module, 
    val_dataloaders: Dict[str, Tuple[DataLoader[Any], int]], 
    training_config: TrainingConfig, 
    accelerator: Any
) -> Dict[str, float]:
    # Setup loss functions
    loss_fn = (
        create_loss_function(training_config.loss) 
        if training_config.loss else torch.nn.CrossEntropyLoss()
    )
    auxiliary_losses = (
        create_auxiliary_losses(training_config.loss) 
        if training_config.loss else {}
    )
    auxiliary_weights = (
        training_config.loss.auxiliary 
        if training_config.loss else {}
    )

    # Run validation manually
    split_metrics = {}
    
    for split_name, (dataloader, _) in val_dataloaders.items():
        total_loss = 0.0
        all_metrics = []
        
        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Validation[{split_name}]",
            disable=not accelerator.is_local_main_process,
            total=total_batches,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            with torch.no_grad():
                outputs = model(
                    x=batch["features"],
                    is_target=batch["is_target"],
                    labels=batch["labels"],
                )
                
                targets = batch["labels"]
                
                # Handle output format and slicing (ignore first token/enrollment)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
                logits = logits[:, 1:, :]

                loss_dict = compute_loss(
                    loss_fn,
                    logits,
                    targets,
                    auxiliary_losses=auxiliary_losses,
                    auxiliary_weights=auxiliary_weights,
                    model=model,
                )
                
                # Compute metrics
                batch_metrics = compute_metrics(
                    logits,
                    targets,
                    task_type="classification",
                )
                
                # Add losses to metrics
                batch_metrics["loss"] = loss_dict["total"].item()
                for k, v in loss_dict.items():
                    if k != "total":
                        batch_metrics[f"loss_{k}"] = v.item()
                        
                all_metrics.append(batch_metrics)

        # Aggregate split metrics
        if not all_metrics:
            split_results = {}
        else:
            keys = all_metrics[0].keys()
            split_results = {}
            for k in keys:
                values = [m[k] for m in all_metrics]
                split_results[k] = sum(values) / len(values)

        # All reduce for distributed
        split_results = all_reduce_metrics(accelerator, split_results)
        split_metrics[split_name] = split_results
        
        if accelerator.is_local_main_process:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in split_results.items())
            print(f"Validation[{split_name}] | {metrics_str}")

    # Aggregate across splits
    metrics = {}
    if split_metrics:
        numeric_keys = None
        for m in split_metrics.values():
            keys = {k for k, v in m.items() if isinstance(v, numbers.Number)}
            if numeric_keys is None:
                numeric_keys = keys
            else:
                numeric_keys &= keys
        
        if numeric_keys:
            for k in numeric_keys:
                metrics[k] = float(np.mean([m[k] for m in split_metrics.values()]))

    if accelerator.is_local_main_process:
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(f"Validation (avg) | {metrics_str}")
        
    return metrics


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

            metrics = validate_model(model, val_dataloaders, training_config, accelerator)
            
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
