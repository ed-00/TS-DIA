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
    "configs/FINETUNE/softmax_linear_model/softmax_linear_model_combined.yaml",
    "configs/FINETUNE/softmax_linear_model_correct_nbf/softmax_linear_model_correct_nbf_combined.yaml" ,
    "configs/FINETUNE/softmax_pretraining_model/softmax_pretraining_model_combined.yaml"
    # "configs/REPRODUCE_eval_only/02_pretraining_softmax.yaml",
    # "configs/REPRODUCE_eval_only/03_pretraining_linear.yaml",
    # "configs/REPRODUCE_eval_only/04_pretraining_linear_correct_nb-f.yaml"
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

        # Add per-split metrics
        for split_name, split_results in split_metrics.items():
            for k, v in split_results.items():
                if isinstance(v, numbers.Number):
                    metrics[f"{split_name}/{k}"] = v

    if accelerator.is_local_main_process:
        # Filter out per-split metrics for the average print
        avg_metrics = {k: v for k, v in metrics.items() if "/" not in k}
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items())
        print(f"Validation (avg) | {metrics_str}")
        
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recursive_checkpoints", action="store_true", help="Recursively search for checkpoints in save_dir")
    parser.add_argument(
        "--ckpt",
        action="append",
        help=(
            "Checkpoint selection mapping in the form 'config_basename:step1,step2' or 'config_basename:step'. "
            "Can be provided multiple times. If provided and --recursive_checkpoints is NOT set, the script will evaluate only the listed checkpoint steps for the matching config."
        ),
    )
    args = parser.parse_args()

    # Initialize results list
    results = []

    # Initialize wandb
    wandb.init(project="ego_evaluation", name="eval_all_models_checkpoints")

    # Parse ckpt mappings once before processing all configs and keep track of used mappings
    ckpt_map = {}
    used_ckpt_keys = set()
    if args.ckpt:
        for mapping in args.ckpt:
            try:
                key, steps = mapping.split(":", 1)
                steps_list = [int(s.strip()) for s in steps.split(",") if s.strip()]
                ckpt_map[key.strip()] = steps_list
            except Exception:
                print(f"Warning: Could not parse checkpoint mapping '{mapping}'. Expected format 'config:step1,step2'. Skipping this mapping.")

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

        cut_sets = data_manager.load_datasets(
            datasets=dataset_configs,
            global_config=train_global_config,
            training_dataset_mapping=TrainingDatasetMap(splits=[]), 
            validation_dataset_mapping=val_mapping,
            strict_splits=True
        )

        label_type = "ego"

        val_dataloaders = data_manager.create_validation_dataloaders(
            cut_sets=cut_sets,
            dataset_configs=dataset_configs,
            global_config=train_global_config,
            training_config=training_config,
            accelerator=accelerator,
            label_type=label_type
        )

        # Prepare dataloaders with accelerator
        for key, (dataloader, size) in val_dataloaders.items():
            val_dataloaders[key] = (accelerator.prepare(dataloader), size)

        if not val_dataloaders:
            print("No validation dataloaders created. Check dataset config and splits.")
            continue
        if not training_config.checkpoint:
            raise ValueError("training_config.checkpoint: was not provided in the config...")
        
        # Find Checkpoints
        # ckpt_map was parsed before the loop; use it here
        save_dir = training_config.checkpoint.save_dir
        if args.recursive_checkpoints:
            checkpoints = glob.glob(os.path.join(save_dir, "**", "checkpoint*"), recursive=True)
        else:
            checkpoints = glob.glob(os.path.join(save_dir, "checkpoint*"))

        checkpoints.sort(key=get_step)

        # If not running recursive checkpoint search and the user provided mappings,
        # use them to filter which checkpoints to evaluate for each config.
        config_basename = os.path.basename(config_path)
        config_basename_noext = os.path.splitext(config_basename)[0]
        desired_steps = None
        if not args.recursive_checkpoints and ckpt_map:
            # Accept matches keyed by either the basename with or without extension
            if config_basename in ckpt_map:
                desired_steps = ckpt_map[config_basename]
                used_ckpt_keys.add(config_basename)
            elif config_basename_noext in ckpt_map:
                desired_steps = ckpt_map[config_basename_noext]
                used_ckpt_keys.add(config_basename_noext)
            else:
                # Also allow user to specify the path suffix (e.g., directories included)
                # Check each key to see if it is contained in the config_path
                for k, v in ckpt_map.items():
                    if k in config_path:
                        desired_steps = v
                        used_ckpt_keys.add(k)
                        break
                # Finally, allow matching on the model_config.name in case the user used that
                if desired_steps is None and hasattr(model_config, "name"):
                    if model_config.name in ckpt_map:
                        desired_steps = ckpt_map[model_config.name]
                        used_ckpt_keys.add(model_config.name)

        if desired_steps is not None:
            filtered_checkpoints = [p for p in checkpoints if get_step(p) in desired_steps]
            if filtered_checkpoints:
                checkpoints = filtered_checkpoints
                print(f"Filtering checkpoints for {config_basename} to steps {desired_steps} -> found {len(checkpoints)} matching checkpoint(s)")
            else:
                print(f"Warning: No checkpoints matching steps {desired_steps} found in {save_dir} for {config_basename}. Skipping this config.")
                continue

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
            model = accelerator.prepare(model)

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

    # Report any unused checkpoint mappings provided by the user
    if ckpt_map:
        unused_keys = set(ckpt_map.keys()) - used_ckpt_keys
        if unused_keys:
            print(f"Warning: The following checkpoint mappings were provided but not used: {list(unused_keys)}")


if __name__ == "__main__":
    main()
