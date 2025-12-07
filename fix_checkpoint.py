import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
import sys

# Add workspace root to sys.path
sys.path.append(str(Path(__file__).parent))

from model.model_factory import ModelFactory
from model.parse_model_args import parse_model_config
from utility.collect_model import collect_model

def fix_checkpoint(config_path, checkpoint_path, output_path):
    print(f"Loading config from {config_path}")
    model_config = parse_model_config(Path(config_path))
    
    print("Creating model structure...")
    model = ModelFactory.create_model(model_config)
    
    print(f"Loading and fixing checkpoint from {checkpoint_path} using collect_model...")
    model = collect_model(checkpoint_path, model, device='cpu')
    
    print(f"Saving fixed checkpoint to {output_path}")
    from safetensors.torch import save_model
    save_model(model, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to input checkpoint")
    parser.add_argument("--output", type=str, default="fixed_model.safetensors", help="Path to output checkpoint")
    args = parser.parse_args()
    
    fix_checkpoint(args.config, args.checkpoint, args.output)
