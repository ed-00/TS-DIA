#!/usr/bin/env python3
"""
Verify training dataloader and computed steps per GPU against actual yields.

Usage:
    PYTHONPATH=$PWD python3 scripts/verify_train_dataloader.py --config configs/REPRODUCE/02_pretraining_softmax.yaml --full-run

This script:
 - Loads config
 - Uses DatasetManager to create training dataloader
 - Prints metadata-derived total training size
 - Computes total examples from combined cuts
 - Prints accelerator/distributed world size
 - Optionally iterates over the dataloader to count yielded examples and compare
"""
import argparse
from pathlib import Path
import math
import yaml
import importlib

from accelerate import Accelerator
from data_manager.data_manager import DatasetManager
from parse_args import unified_parser
from training.ego_dataset import EgoCentricDiarizationDataset
from lhotse import CutSet
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--full-run', action='store_true', help='Iterate entire dataloader (slow)')
    parser.add_argument('--num-batches', type=int, default=10, help='Number of batches to inspect when not full run')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration via training.parse args helper
    # The unified_parser reads CLI args itself, but we can set argv before calling (little hack)
    import sys
    sys_argv_backup = list(sys.argv)
    sys.argv = [sys.argv[0], '--config', str(args.config)]
    _, model_cfg, dataset_cfgs, global_cfg, training_cfg, cfg_path = unified_parser()
    sys.argv = sys_argv_backup

    accelerator = Accelerator()

    dm = DatasetManager()
    print('Loading datasets (this will use existing caches)')
    if dataset_cfgs is None or global_cfg is None or training_cfg is None or training_cfg.training_dataset_map is None:
        raise SystemExit('Missing dataset or training config in provided config file')
    cut_sets = dm.load_datasets(datasets=dataset_cfgs, global_config=global_cfg, training_dataset_mapping=training_cfg.training_dataset_map, cache_dir=global_cfg.cache_dir)

    # Create training dataloader
    train_dataloader, total_size = dm.create_training_dataloader(
        cut_sets=cut_sets,
        global_config=global_cfg,
        training_config=training_cfg,
        accelerator=accelerator,
    )

    print('Returned total size from create_training_dataloader:', total_size)

    # Compute combined *actual* examples from combined cuts
    train_cuts = None
    from utility.dataset_utils import prepare_training_cuts
    train_cuts = prepare_training_cuts(cut_sets, training_cfg.training_dataset_map)
    print('len(train_cuts) (num cuts after windowing):', len(train_cuts))

    computed_total = EgoCentricDiarizationDataset.get_total_dataset_size(train_cuts, desc='Computing combined examples for train')
    print('Computed total examples from combined cuts:', computed_total)

    # Print accelerator info
    print('Accelerator num_processes:', accelerator.num_processes)
    try:
        import torch.distributed as dist
        print('torch.distributed initialized:', dist.is_initialized())
        if dist.is_initialized():
            print('torch.distributed world_size:', dist.get_world_size())
            print('torch.distributed rank:', dist.get_rank())
    except Exception as exc:
        print('Error checking torch.distributed:', exc)

    # Compute expected steps
    examples_per_process = math.ceil(total_size / max(1, accelerator.num_processes))
    steps_per_gpu_expected = math.ceil(examples_per_process / (training_cfg.batch_size * training_cfg.gradient_accumulation_steps))
    print(f'Computed from metadata: examples_per_process={examples_per_process} steps_per_gpu_expected={steps_per_gpu_expected}')

    # Optionally iterate dataloader
    if args.full_run:
        print('Full run: iterating entire dataloader...')
        total_seen = 0
        batches = 0
        for batch_idx, batch in enumerate(train_dataloader):
            batches += 1
            total_seen += batch['features'].size(0)
            if batches % 1000 == 0:
                print('  Seen', total_seen, 'in', batches, 'batches')
        print('Full run complete: batches=', batches, 'yielded examples=', total_seen)
    else:
        print('Sample run: iterating', args.num_batches, 'batches')
        total_seen = 0
        batches = 0
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx >= args.num_batches:
                break
            batches += 1
            total_seen += batch['features'].size(0)
        print('Sample run complete: batches=', batches, 'yielded examples=', total_seen)

    print('Done.')


if __name__ == '__main__':
    main()
