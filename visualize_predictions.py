from model.utils import fix_safetensors_shared_parameters
from safetensors.torch import load_file
from training.ego_dataset import EgoCentricDiarizationDataset
from training.config import TrainingDatasetMap
from data_manager.parse_args import parse_dataset_configs
from data_manager import GlobalConfig
from data_manager.data_manager import DatasetManager
from model.parse_model_args import parse_model_config
from model.model_factory import ModelFactory
import argparse
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
from tqdm import tqdm
from lhotse import CutSet
# Add workspace root to sys.path
sys.path.append(str(Path(__file__).parent))





def plot_results(features, ref_activity, hyp_activity, output_path):
    """
    features: (T, F) numpy array
    ref_activity: (T, NumRefSpk) numpy array (boolean or 0/1)
    hyp_activity: (T, NumHypSpk) numpy array (probabilities or 0/1)
    """
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot Features
    # Transpose to (F, T) for imshow
    im0 = axes[0].imshow(features.T, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Features (MFCC - Spliced & Subsampled)")
    axes[0].set_ylabel("Dimension")
    plt.colorbar(im0, ax=axes[0])

    # Plot Reference
    # Transpose to (NumSpk, T)
    im1 = axes[1].imshow(ref_activity.T, aspect='auto', origin='lower',
                         cmap='binary', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Speaker Activity")
    axes[1].set_ylabel("Speaker Index")
    axes[1].set_yticks(range(ref_activity.shape[1]))
    cbar1 = plt.colorbar(im1, ax=axes[1], ticks=[0, 1])
    cbar1.ax.set_yticklabels(['Inactive', 'Active'])

    # Plot Hypothesis
    # Transpose to (NumSpk, T)
    im2 = axes[2].imshow(hyp_activity.T, aspect='auto', origin='lower',
                         cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    axes[2].set_title("Predicted Speaker Activity (Binary)")
    axes[2].set_ylabel("Speaker Index")
    axes[2].set_xlabel("Frame Index (Subsampled)")
    axes[2].set_yticks(range(hyp_activity.shape[1]))
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=Path,
                        required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. simu_2spk)")
    parser.add_argument("--split", type=str, required=True,
                        help="Split name (e.g. dev_b5_mix500)")
    parser.add_argument("--cut_id", type=str, default=None,
                        help="Specific cut ID to visualize")
    parser.add_argument("--output", type=str,
                        default="prediction_viz.png", help="Output image path")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Force CPU if CUDA not available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    # Parse configs
    print(f"Loading config from {args.config}")
    dataset_configs, global_config = parse_dataset_configs(args.config)
    model_config = parse_model_config(args.config)

    # Filter dataset configs to only include the requested one
    filtered_configs = [d for d in dataset_configs if d.name == args.dataset]
    if not filtered_configs:
        print(f"Error: Dataset {args.dataset} not found in config.")
        return
    dataset_configs = filtered_configs

    # Create dummy TrainingDatasetMap
    training_dataset_map = TrainingDatasetMap(combine=False, splits=[])

    # Load datasets
    print("Loading datasets...")

    # # Remove the chunk_size to disable windowing
    # if global_config.data_loading:
    #     global_config.data_loading.chunk_size = 30

    cut_sets = DatasetManager().load_datasets(
        datasets=dataset_configs,
        global_config=global_config,
        training_dataset_mapping=training_dataset_map,
        exclude_splits=["train"]
    )

    if args.dataset not in cut_sets:
        print(
            f"Error: Dataset {args.dataset} not found in loaded datasets: {list(cut_sets.keys())}")
        return

    if args.split not in cut_sets[args.dataset]:
        print(
            f"Error: Split {args.split} not found in dataset {args.dataset}: {list(cut_sets[args.dataset].keys())}")
        return

    cuts = cut_sets[args.dataset][args.split].to_eager()
    print(f"Loaded {len(cuts)} cuts from {args.dataset}/{args.split}")

    # Select cut
    if args.cut_id:
        cuts_list = list(cuts)
        selected_cut = next(
            (c for c in cuts_list if c.id == args.cut_id), None)
        if selected_cut is None:
            print(f"Error: Cut {args.cut_id} not found.")
            return
    else:
        # Pick random cut
        import random
        # Convert to list if it's not (CutSet is iterable)
        cuts_list = list(cuts)
        
        # Try to find a cut with speakers
        selected_cut = None
        found = False
        for _ in range(100):
            candidate_cut = random.choice(cuts_list)
            speakers = set(s.speaker for s in candidate_cut.supervisions if s.speaker)
            if len(speakers) > 0:
                selected_cut = candidate_cut
                found = True
                break
        
        if not found or selected_cut is None:
            print("Warning: Could not find a cut with speakers. Using random cut.")
            selected_cut = random.choice(cuts_list)
            
        print(f"Selected random cut: {selected_cut.id}")

    # Create Model
    print("Creating model...")
    model = ModelFactory.create_model(model_config)
    model.to(args.device)
    model.eval()

    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    if str(args.checkpoint).endswith(".safetensors"):
        state_dict = load_file(args.checkpoint)
        state_dict = fix_safetensors_shared_parameters(state_dict, model)
        model.load_state_dict(state_dict)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    single_cut_set = CutSet.from_cuts([selected_cut])

    context_size = 7
    subsampling = 10
    dataset = EgoCentricDiarizationDataset(
        single_cut_set, context_size=context_size, subsampling=subsampling)
    
    item = next(iter(dataset))

    cut = selected_cut
    features = item["features"].to(args.device)  # (T, F)
    ref_activity = item["ground_truth"].to(args.device)  # (NumSpk, T)

    # Add batch dimension
    features_batch = features.unsqueeze(0)  # (1, T, F)

    print(f"Running inference on cut {cut.id}...")
    print(f"Features shape: {features.shape}")
    print(f"Ref activity shape: {ref_activity.shape}")

    with torch.no_grad():
        outputs = model.estimate(
            features_batch,
            enrollment_strategy='spectral_clustering',
            threshold=0.5,
            debug=True
        )

    # outputs: (1, T, max_spk)
    hyp_activity = outputs[0].cpu().numpy()  # (T, max_spk)
    
    # Ref activity is (NumSpk, T), transpose to (T, NumSpk) for consistency
    ref_activity_np = ref_activity.cpu().numpy().T
    features_np = features.cpu().numpy()
    
    # Filter out silent speakers from hypothesis
    active_hyp_indices = np.where(hyp_activity.sum(axis=0) > 0)[0]
    hyp_activity = hyp_activity[:, active_hyp_indices]
    
    # Filter out silent speakers from reference
    active_ref_indices = np.where(ref_activity_np.sum(axis=0) > 0)[0]
    ref_activity_np = ref_activity_np[:, active_ref_indices]

    print(f"Hyp activity shape: {hyp_activity.shape}")

    print(f"Features (MFCC) shape: {features_np.shape}")
    print(f"Ref activity (Labels) shape: {ref_activity_np.shape}")
    print(f"Hyp activity (Predictions) shape: {hyp_activity.shape}")

    plot_results(features_np, ref_activity_np, hyp_activity, args.output)


if __name__ == "__main__":
    main()
