import glob
import os
import re

save_dirs = [
    "./outputs/checkpoints/softmax_pretraining_large_pos_loss_patch_final_restart_clean_no_rezero/checkpoints",
    "./outputs/checkpoints/linear_pretraining_large_pos_loss_patch_final_clean_no_rezero/checkpoints",
    "./outputs/checkpoints/linear_pretraining_large_pos_loss_patch_final_clean_no_rezero_nbf25/checkpoints"
]

for save_dir in save_dirs:
    print(f"\nChecking {save_dir}")
    if not os.path.exists(save_dir):
        print(f"Directory does not exist: {save_dir}")
        continue
        
    # Non-recursive
    checkpoints = glob.glob(os.path.join(save_dir, "checkpoint*"))
    print(f"Found {len(checkpoints)} checkpoints (non-recursive)")
    if len(checkpoints) > 0:
        print(f"Sample: {checkpoints[0]}")

    # Recursive
    checkpoints_rec = glob.glob(os.path.join(save_dir, "**", "checkpoint*"), recursive=True)
    print(f"Found {len(checkpoints_rec)} checkpoints (recursive)")
    if len(checkpoints_rec) > 0:
        print(f"Sample: {checkpoints_rec[0]}")
        
    # Sort logic check
    def get_step(path):
        name = os.path.basename(path)
        match = re.search(r'(\d+)$', name)
        if match:
            return int(match.group(1))
        return 0
    
    checkpoints.sort(key=get_step)
    if len(checkpoints) > 0:
        print(f"First sorted: {checkpoints[0]}")
        print(f"Last sorted: {checkpoints[-1]}")
