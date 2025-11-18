#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import yaml
from lhotse import CutSet
from training.ego_dataset import EgoCentricDiarizationDataset

def load_yaml(path: Path):
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)


def main(config_path: Path, cache_root: Path):
    cfg = load_yaml(config_path)
    training_map = cfg.get('training', {}).get('training_dataset_map', None)
    if training_map is None:
        print('No training map found in config')
        return
    splits = training_map.get('splits', [])
    total_cached = 0
    total_cached_adjusted = 0
    total_combined = 0

    combined_cuts = []
    for sp in splits:
        dataset_name = sp['dataset_name']
        split_name = sp['split_name']
        subset_ratio = sp.get('subset_ratio', 1.0)
        meta_path = cache_root / dataset_name / split_name / 'cache_metadata.json'
        cuts_path = cache_root / dataset_name / split_name / 'cuts_windowed.jsonl.gz'
        if meta_path.exists():
            meta = json.load(open(meta_path))
            dataset_size_meta = meta.get('dataset_size', None)
            print(f"{dataset_name}/{split_name} cached dataset_size={dataset_size_meta}")
            total_cached += dataset_size_meta if dataset_size_meta else 0
            subset_ratio = sp.get('subset_ratio', 1.0)
            if subset_ratio and 0 < subset_ratio < 1.0:
                total_cached_adjusted += int(dataset_size_meta * subset_ratio)
            else:
                total_cached_adjusted += dataset_size_meta if dataset_size_meta else 0
        else:
            print(f"{dataset_name}/{split_name} has no metadata")
        if cuts_path.exists():
            cuts = CutSet.from_jsonl_lazy(cuts_path)
            # apply subset ratio if set (subset first N cuts)
            if 0 < subset_ratio < 1.0:
                cuts = cuts.subset(first=int(len(cuts)*subset_ratio))
            combined_cuts.append(cuts)

    if combined_cuts:
        if len(combined_cuts) == 1:
            all_cuts = combined_cuts[0]
        else:
            from lhotse.cut import CutSet as _CutSet
            all_cuts = _CutSet.mux(*combined_cuts)
        combined_total_examples = EgoCentricDiarizationDataset.get_total_dataset_size(all_cuts, desc='calc combined')
        print(f"Combined cuts total examples (computed): {combined_total_examples}")
        print(f"Total cached sum across splits: {total_cached}")
        print(f"Total cached sum adjusted by subset_ratio: {total_cached_adjusted}")

        # Compute expected steps per pprocess for given training config if present
        training_cfg = cfg.get('training', {})
        batch_size = training_cfg.get('batch_size', 256)
        grad_acc = training_cfg.get('gradient_accumulation_steps', 1)
        world_size = 1  # default assume 1; user can pass as arg if needed
        per_process_examples = int(total_cached_adjusted / max(1, world_size))
        steps_per_gpu = (per_process_examples + (batch_size*grad_acc) - 1) // (batch_size*grad_acc)
        print(f"Expected steps per process (computed from cached adjusted): {steps_per_gpu} (per_process_examples={per_process_examples})")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: diagnose_dataset_sizes.py <config.yml> <cache_root>')
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
