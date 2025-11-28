import argparse
import torch
import numpy as np
from pathlib import Path
from accelerate import Accelerator
from tqdm import tqdm
import yaml
import sys
from torch.utils.data import Dataset, DataLoader

# Add workspace root to sys.path
sys.path.append(str(Path(__file__).parent))

from model.model_factory import ModelFactory
from model.parse_model_args import parse_model_config
from data_manager.data_manager import DatasetManager
from data_manager import GlobalConfig
from data_manager.parse_args import parse_dataset_configs
from training.config import TrainingDatasetMap
from utility.metrics import compute_der
from training.ego_dataset import splice, subsample_torch

class DiarizationDataset(Dataset):
    """
    Standard Diarization Dataset that yields features and the cut.
    Not ego-centric (i.e. does not select a specific target speaker).
    """
    def __init__(self, cuts, context_size=7, subsampling=10, min_speaker_dim=None):
        self.cuts = list(cuts)
        self.context_size = context_size
        self.subsampling = subsampling
        self.min_speaker_dim = min_speaker_dim

    def __len__(self):
        return len(self.cuts)

    def __getitem__(self, idx):
        cut = self.cuts[idx]
        # load_features returns numpy array (T, F)
        features = torch.from_numpy(cut.load_features())
        
        # Get speaker activity mask
        # We need to determine the speakers for this cut
        all_speakers = sorted(list(set(s.speaker for s in cut.supervisions if s.speaker)))
        speaker_to_idx_map = {spk: i for i, spk in enumerate(all_speakers)}
        
        # speakers_feature_mask returns (num_speakers, num_frames)
        speaker_activity = torch.from_numpy(cut.speakers_feature_mask(
            min_speaker_dim=self.min_speaker_dim,
            speaker_to_idx_map=speaker_to_idx_map
        ))
        
        # Apply subsampling
        if self.subsampling > 1:
            # subsample_torch expects (T, ...) for both inputs
            # speaker_activity is (num_speakers, T), so we transpose
            features, speaker_activity_T = subsample_torch(
                features, 
                speaker_activity.transpose(0, 1), 
                subsample=self.subsampling
            )
            # Transpose back to (num_speakers, T)
            speaker_activity = speaker_activity_T.transpose(0, 1)
            
        # Apply splicing
        features = splice(features, context_size=self.context_size)
        
        return {
            "features": features, 
            "cut": cut,
            "speaker_activity": speaker_activity
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-only", action="store_true", help="Skip training splits")
    args = parser.parse_args()
    
    # Force CPU if CUDA not available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    # Parse configs
    print(f"Loading config from {args.config}")
    dataset_configs, global_config = parse_dataset_configs(args.config)
    model_config = parse_model_config(args.config)
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Initialize DatasetManager
    data_manager = DatasetManager()
    
    # # Disable windowing for evaluation
    # if global_config.data_loading:
    #     global_config.data_loading.chunk_size = None
    
    # Create dummy TrainingDatasetMap
    training_dataset_map = TrainingDatasetMap(combine=False, splits=[])
    
    # Load datasets
    print("Loading datasets...")
    # We assume the config has 'datasets' section which are test datasets
    # load_datasets returns Dict[str, Dict[str, CutSet]]
    # The keys are dataset names, values are dicts of splits (train, val, test)
    exclude_splits = ["train"] if args.test_only else None
    cut_sets = data_manager.load_datasets(
        datasets=dataset_configs,
        global_config=global_config,
        training_dataset_mapping=training_dataset_map,
        exclude_splits=exclude_splits
    )
    
    # Create Model
    print("Creating model...")
    model = ModelFactory.create_model(model_config)
    model.to(args.device)
    model.eval()
    
    # Load Checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        if str(args.checkpoint).endswith(".safetensors"):
            from safetensors.torch import load_file
            from model.utils import fix_safetensors_shared_parameters
            
            state_dict = load_file(args.checkpoint)
            state_dict = fix_safetensors_shared_parameters(state_dict, model)
            
            model.load_state_dict(state_dict)
        else:
            checkpoint = torch.load(args.checkpoint, map_location=args.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
    else:
        print("WARNING: No checkpoint provided. Using random weights.")

    # Iterate over datasets
    results = {}
    i = 0
    for dataset_name, splits in cut_sets.items():

        print(f"Processing dataset: {dataset_name}")
        for split_name, cuts in splits.items():
            if args.test_only and "train" in split_name:
                continue
            print(f"  Split: {split_name}, Cuts: {len(cuts)}")
            if i == 0:
                i += 1
                continue
            total_der = 0.0
            total_speech = 0.0
            count = 0
            
            # Create dataset and dataloader
            context_size = 7
            subsampling = 10
            dataset = DiarizationDataset(cuts, context_size=context_size, subsampling=subsampling)
            dataloader = DataLoader(
                dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=0,
                collate_fn=lambda x: x[0]
            )
            
            pbar = tqdm(dataloader, desc=f"{dataset_name}/{split_name}")
            for batch in pbar:
                cut = batch["cut"]
                try:
                    features = batch["features"].to(args.device)
                except Exception as e:
                    print(f"    Error loading features for cut {cut.id}: {e}")
                    continue
                
                # Add batch dimension
                features = features.unsqueeze(0) # (1, T, F)
                
                # Run estimate
                with torch.no_grad():
                    # estimate returns (batch, seq_len, max_spk)
                    # We need to pass enrollment_strategy etc.
                    # Using defaults for now or from config if available?
                    # Config doesn't seem to have inference params.
                    outputs = model.estimate(
                        features,
                        enrollment_strategy='spectral_clustering', # Good default
                        threshold=0.5
                    )
                    
                # Process outputs
                # outputs: (1, T, max_spk)
                activity = outputs[0].cpu().numpy() > 0.5 # (T, max_spk)
                
                # Convert to segments
                hyp_segments = []
                num_frames, num_spk = activity.shape
                frame_shift = 0.01 * subsampling
                
                for spk_idx in range(num_spk):
                    # Find continuous segments
                    is_active = activity[:, spk_idx]
                    # Simple run-length encoding
                    diff = np.diff(np.concatenate(([0], is_active.astype(int), [0])))
                    starts = np.where(diff == 1)[0]
                    ends = np.where(diff == -1)[0]
                    for s, e in zip(starts, ends):
                        hyp_segments.append((s * frame_shift, e * frame_shift, f"hyp_{spk_idx}"))
                        
                # Get reference segments
                ref_segments = []
                for supervision in cut.supervisions:
                    ref_segments.append((supervision.start, supervision.end, supervision.speaker))
                    
                # Compute DER
                metrics = compute_der(ref_segments, hyp_segments, cut.duration)
                
                # Weighted average by speech duration
                total_der += metrics["DER"] * metrics["TotalSpeech"]
                total_speech += metrics["TotalSpeech"]
                count += 1
                
                current_avg_der = total_der / total_speech if total_speech > 0 else 0.0
                pbar.set_postfix(DER=f"{current_avg_der*100:.2f}%")
                
            avg_der = total_der / total_speech if total_speech > 0 else 0.0
            print(f"  Result {dataset_name}/{split_name}: DER = {avg_der*100:.2f}%")
            results[f"{dataset_name}/{split_name}"] = avg_der

    print("\nFinal Results:")
    for k, v in results.items():
        print(f"{k}: DER = {v*100:.2f}%")

if __name__ == "__main__":
    main()
