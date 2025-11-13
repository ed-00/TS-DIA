#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Dataset Utility Functions

This module provides utility functions for manipulating and preparing datasets,
specifically for combining and splitting `CutSet` objects from Lhotse.
"""
from typing import Dict

from lhotse import CutSet

from training.config import TrainingDatasetMap


def prepare_training_cuts(
    cut_sets: Dict[str, Dict[str, CutSet]],
    training_dataset_map: TrainingDatasetMap,
) -> CutSet:
    """
    Prepare the training cuts based on the training_dataset_map.

    This function processes a dictionary of CutSets according to the
    specifications in a TrainingDatasetMap object. It supports taking
    subsets of specified dataset splits and combining them into a single
    CutSet for training.

    Args:
        cut_sets: A dictionary where keys are dataset names and values are
                  dictionaries mapping split names to CutSet objects.
                  Example: {'ami': {'train': CutSet(...), 'dev': CutSet(...)}}
        training_dataset_map: A configuration object that specifies which
                              dataset splits to use, what subset ratio to
                              apply, and whether to combine them.

    Returns:
        A single CutSet containing the combined and subsetted training data.

    Raises:
        ValueError: If a specified dataset or split is not found in `cut_sets`.
    """
    if not training_dataset_map.splits:
        raise ValueError("training_dataset_map contains no splits.")

    all_train_cuts = []
    for split_info in training_dataset_map.splits:
        dataset_name = split_info.dataset_name
        split_name = split_info.split_name
        subset_ratio = split_info.subset_ratio

        if dataset_name not in cut_sets:
            raise ValueError(f"Dataset '{dataset_name}' not found in loaded cut_sets.")

        dataset_cuts = cut_sets[dataset_name]
        if split_name not in dataset_cuts:
            available_splits = ", ".join(dataset_cuts.keys())
            raise ValueError(
                f"Split '{split_name}' not found for dataset '{dataset_name}'. "
                f"Available splits: {available_splits}"
            )

        target_cuts = dataset_cuts[split_name]

        # Apply subsetting based on the ratio
        if 0 < subset_ratio < 1.0:
            num_cuts = int(len(target_cuts) * subset_ratio)
            target_cuts = target_cuts.subset(first=num_cuts)
            print(
                f"  - Using {subset_ratio:.2%} ({num_cuts} cuts) of {dataset_name}/{split_name}"
            )
        else:
            print(f"  - Using all cuts from {dataset_name}/{split_name}")

        all_train_cuts.append(target_cuts)

    if training_dataset_map.combine and len(all_train_cuts) > 1:
        print("\nCombining training splits into a single dataset.")
        combined_cuts = CutSet.mux(*all_train_cuts)
        print(f"Total combined cuts: {len(combined_cuts)}")
        return combined_cuts
    else:
        # If not combining or only one split, return the first (or only) one
        return all_train_cuts[0]
