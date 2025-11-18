#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
from lhotse import CutSet
from training.ego_dataset import EgoCentricDiarizationDataset


def main(cache_root: Path, dataset_name: str, split_name: str, limit: int = 1000):
    cache_dir = cache_root / dataset_name / split_name
    metadata_file = cache_dir / "cache_metadata.json"
    cuts_file = cache_dir / "cuts_windowed.jsonl.gz"

    if not metadata_file.exists():
        print(f"No metadata at {metadata_file}")
        return
    if not cuts_file.exists():
        print(f"No cuts file at {cuts_file}")
        return

    meta = json.load(open(metadata_file))
    print(f"Metadata for {dataset_name}/{split_name}: {meta}")

    cuts = CutSet.from_jsonl_lazy(cuts_file)
    print(f"Len cuts: {len(cuts)}")

    dataset = EgoCentricDiarizationDataset(cuts=cuts)
    count = 0
    per_cut_counts = []
    for i, cut in enumerate(cuts):
        # Count speakers per cut
        all_speakers_in_cut = sorted(
            set(s.speaker for s in cut.supervisions if s.speaker)
        )
        expected_count = len(all_speakers_in_cut) + 1
        per_cut_counts.append(expected_count)
        if i + 1 >= limit:
            break

    avg_expected = sum(per_cut_counts)/len(per_cut_counts) if per_cut_counts else 0
    print(f"Sampled {len(per_cut_counts)} cuts. Avg expected examples per cut (speakers+1): {avg_expected:.2f}")

    # Now actually iterate dataset limited number of examples and count per cut
    # Because dataset yields in cut order, we can map yields per cut using generator
    cut_yield_counts = {}
    iterator = iter(dataset)
    samples_seen = 0
    # Map cut.id to counts
    for n in range(limit * 2):  # a safe upper bound
        try:
            item = next(iterator)
        except StopIteration:
            break
        samples_seen += 1
    print(f"Yielded {samples_seen} examples after {limit*2} attempts")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-root', default='cache', type=Path)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    main(args.cache_root, args.dataset, args.split, args.limit)
