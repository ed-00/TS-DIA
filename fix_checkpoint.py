import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
import sys

# Add workspace root to sys.path
sys.path.append(str(Path(__file__).parent))

from model.model_factory import ModelFactory
from model.parse_model_args import parse_model_config

def fix_checkpoint(config_path, checkpoint_path, output_path):
    print(f"Loading config from {config_path}")
    model_config = parse_model_config(Path(config_path))
    
    print("Creating model structure...")
    model = ModelFactory.create_model(model_config)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = load_file(checkpoint_path)
    
    print("Fixing state_dict for shared parameters...")
    model_state_dict = model.state_dict()
    
    # Map data_ptr to list of keys in the model
    ptr_to_keys = {}
    for k, v in model_state_dict.items():
        ptr = v.data_ptr()
        if ptr not in ptr_to_keys:
            ptr_to_keys[ptr] = []
        ptr_to_keys[ptr].append(k)
    
    # Fill in missing keys
    fixed_state_dict = dict(state_dict)
    fixed_count = 0
    
    for k in model_state_dict.keys():
        if k not in fixed_state_dict:
            # Find a sibling key that might be in the state_dict
            ptr = model_state_dict[k].data_ptr()
            siblings = ptr_to_keys.get(ptr, [])
            
            found = False
            for sibling in siblings:
                if sibling in fixed_state_dict:
                    fixed_state_dict[k] = fixed_state_dict[sibling]
                    fixed_count += 1
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find value for missing key {k}")

    print(f"Restored {fixed_count} missing shared keys.")
    
    print("Loading fixed state_dict into model...")
    model.load_state_dict(fixed_state_dict)
    
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
