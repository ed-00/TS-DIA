import glob
import os

save_dir = "./outputs/checkpoints/linear_pretraining_large_pos_loss_patch_final_clean_no_rezero_nbf25/checkpoints"
print(f"Checking {save_dir}")

checkpoints = glob.glob(os.path.join(save_dir, "**", "checkpoint-*"), recursive=True)
print(f"Found {len(checkpoints)} checkpoints")
for c in checkpoints[:5]:
    print(c)
