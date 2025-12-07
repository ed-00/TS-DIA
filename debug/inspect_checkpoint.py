import torch
from safetensors.torch import load_file
import sys

checkpoint_path = "outputs/checkpoints/softmax_pretraining_large_pos_loss_patch_final_restart_clean_no_rezero/checkpoints/checkpoint_88/model.safetensors"

try:
    state_dict = load_file(checkpoint_path)
    print("Keys in checkpoint:")
    for key in state_dict.keys():
        print(key)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
