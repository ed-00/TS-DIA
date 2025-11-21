#!/usr/bin/env python3
"""
Validate EgoCentricDiarizationDataset and DataLoader behavior for a given cached cutset.

Usage:
    PYTHONPATH=$PWD python3 scripts/validate_ego_dataloader.py \
      --cache-root /workspace/cache --dataset simu_1spk --split train_b2_mix100000 \
      --batch-size 256 --num-batches 10 --num-workers 4

The script prints sample batch shapes and some simple sanity checks.
"""
import argparse
from pathlib import Path
import math
import json

import torch
from torch.utils.data import DataLoader
from lhotse import CutSet

from training.ego_dataset import EgoCentricDiarizationDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', type=Path, default=Path('cache'))
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-batches', type=int, default=10)
    parser.add_argument('--context-size', type=int, default=7)
    parser.add_argument('--subsampling', type=int, default=10)
    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--shuffle', action='store_true', help='Use dataset shuffle buffer if available')
    parser.add_argument('--compute-size', action='store_true', help='Compute expected total examples from the CutSet')
    parser.add_argument('--full-run', action='store_true', help='Iterate entire dataloader to count all examples (slow)')
    return parser.parse_args()


def main():
    args = parse_args()

    cache_dir = Path(args.cache_root) / args.dataset / args.split
    cuts_file = cache_dir / 'cuts_windowed.jsonl.gz'
    meta_file = cache_dir / 'cache_metadata.json'

    if not cuts_file.exists():
        raise SystemExit(f"Cutset file not found: {cuts_file}")

    print(f"Loading cuts from: {cuts_file}")
    cuts = CutSet.from_jsonl_lazy(cuts_file)
    print(f"Len cuts: {len(cuts)} (num windowed cuts)")

    if meta_file.exists():
        meta = json.load(open(meta_file))
        print(f"Loaded cache metadata: dataset_size={meta.get('dataset_size')}, num_cuts={meta.get('num_cuts')}")
    else:
        meta = None

    dataset = EgoCentricDiarizationDataset(
        cuts=cuts,
        context_size=args.context_size,
        subsampling=args.subsampling,
        shuffle_buffer_size=10000 if args.shuffle else 0,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=EgoCentricDiarizationDataset.collate_fn,
        drop_last=args.drop_last,
        pin_memory=(args.num_workers > 0),
        persistent_workers=(args.num_workers > 0),
    )

    print(f"Created dataloader with batch_size={args.batch_size}, num_workers={args.num_workers}, drop_last={args.drop_last}")

    total_examples = None
    if args.compute_size:
        print("Computing total expected dataset size from cuts (this may be slow)...")
        total_examples = EgoCentricDiarizationDataset.get_total_dataset_size(cuts, desc='Compute dataset total examples')
        print(f"Computed total examples in dataset: {total_examples}")
        expected_batches = math.ceil(total_examples / args.batch_size)
        print(f"Expected batches (no grad_accum considered): {expected_batches}")

    # Iterate for a small number of batches and print diagnostics
    sample_batch = None
    total_seen = 0
    batches_processed = 0

    try:
        if args.full_run:
            print("Full run: iterating entire dataloader (this may take a long time)...")
            batches_processed = 0
            total_seen = 0
            for batch_idx, batch in enumerate(dataloader):
                batches_processed += 1
                x = batch['features']
                y = batch['labels']
                total_seen += x.size(0)
                if batches_processed % 1000 == 0:
                    print(f"  Seen {total_seen} examples in {batches_processed} batches...")
            print(f"Full run complete. Batches processed: {batches_processed}, total examples seen (local): {total_seen}")
        else:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= args.num_batches:
                    break
                batches_processed += 1
                x = batch['features']
                y = batch['labels']
                is_target = batch['is_target']

                print(f"Batch {batch_idx}: features.shape={tuple(x.shape)} labels.shape={tuple(y.shape)} is_target.shape={tuple(is_target.shape)}")

                # Basic sanity checks
                if x.isnan().any():
                    print("  ⚠️  NaNs found in features!")
                if (y < 0).any():
                    print("  ⚠️  Negative label values found (maybe IGNORE_INDEX?)")

                total_seen += x.size(0)

        # Summary output
        if not args.full_run:
            print(f"Batches processed: {batches_processed}, total examples seen (local): {total_seen}")
        # Compute expected counts and comparisons
        if meta is not None:
            meta_size = meta.get('dataset_size')
            print(f"Cached dataset_size (metadata): {meta_size}")
        else:
            meta_size = None
        computed = total_examples if args.compute_size else None

        yielded = total_seen

        # Print comparison table
        print("\nDataset size comparison (examples):")
        print(f"  computed (from cuts): {computed}")
        print(f"  cached metadata:     {meta_size}")
        print(f"  yielded by dataloader: {yielded}")
        # Expected yielded examples considering drop_last
        if computed is not None:
            if args.drop_last:
                expected_yielded = (computed // args.batch_size) * args.batch_size
            else:
                expected_yielded = computed
            print(f"  expected yielded (drop_last={args.drop_last}): {expected_yielded}")
            exp_delta = yielded - expected_yielded
            exp_pct = (exp_delta / expected_yielded * 100.0) if expected_yielded else 0.0
            print(f"  difference yielded - expected_yielded = {exp_delta} ({exp_pct:.2f}%)")
        if computed is not None:
            delta = yielded - computed
            pct = (delta / computed * 100.0) if computed else 0.0
            print(f"  difference yielded - computed = {delta} ({pct:.2f}%)")

    except Exception as exc:
        print(f"Error while iterating dataloader: {exc}")
        raise


if __name__ == '__main__':
    main()
