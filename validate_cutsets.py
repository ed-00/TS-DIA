#! /usr/bin/env python
"""Utility to load cached manifests and print CutSet.describe() per split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_manager.data_manager import DatasetManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load existing dataset manifests and print CutSet.describe information."
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("./outputs/manifests"),
        help="Directory that contains per-dataset manifest subfolders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset names to validate. Defaults to all subdirectories.",
    )
    return parser.parse_args()


def discover_datasets(base_dir: Path) -> list[str]:
    if not base_dir.exists():
        return []
    return sorted(
        entry.name for entry in base_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )


def validate_dataset(dataset_name: str, manifests_dir: Path) -> bool:
    dataset_path = manifests_dir / dataset_name
    if not dataset_path.exists():
        print(f"✗ {dataset_name}: manifest directory not found at {dataset_path}")
        return False

    manifests = DatasetManager._try_load_existing_manifests(dataset_path, dataset_name)
    if not manifests:
        print(f"✗ {dataset_name}: no manifests detected in {dataset_path}")
        return False

    try:
        cut_sets = DatasetManager._manifests_to_cutsets_dict(manifests, dataset_name)
    except Exception as exc:  
        print(f"✗ {dataset_name}: failed to convert manifests to CutSets ({exc})")
        return False

    if not cut_sets:
        print(f"✗ {dataset_name}: manifests converted to empty CutSet dictionary")
        return False

    print(f"\n✅ {dataset_name}")
    for split_name, cut_set in cut_sets.items():
        print(f"--- split: {split_name}")
        cut_set.describe()
    return True


def main() -> int:
    args = parse_args()
    manifests_dir = args.manifests_dir.resolve()

    dataset_names = args.datasets or discover_datasets(manifests_dir)
    if not dataset_names:
        print("No datasets to validate.")
        return 1

    overall_success = True
    for dataset_name in dataset_names:
        success = validate_dataset(dataset_name, manifests_dir)
        overall_success &= success

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
