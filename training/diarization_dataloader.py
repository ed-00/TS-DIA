#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Diarization DataLoader Creation using Lhotse

This module provides utilities for creating PyTorch DataLoaders from Lhotse CutSets
for speaker diarization tasks using Lhotse's built-in capabilities.

Key Functions:
    create_diarization_dataloader: Create DataLoader from CutSet for diarization
    create_train_val_dataloaders: Create training and validation DataLoaders
"""

from typing import Optional, Tuple

from lhotse import CutSet, Fbank, FbankConfig, Mfcc, MfccConfig
from lhotse.dataset import BucketingSampler, DiarizationDataset, SimpleCutSampler
from torch.utils.data import DataLoader, IterableDataset


class LhotseIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset that wraps Lhotse samplers.

    This bypasses PyTorch's indexing mechanism entirely and lets the Lhotse sampler
    handle all the iteration logic. This is the correct way to integrate Lhotse
    with PyTorch DataLoader.
    """

    def __init__(self, sampler, dataset):
        self.sampler = sampler
        self.dataset = dataset

    def __iter__(self):
        """Iterate through the sampler and yield dataset results."""
        for batch_cuts in self.sampler:
            yield self.dataset[batch_cuts]

    def __len__(self):
        """Return the precomputed number of batches."""
        return getattr(self.sampler, "_num_batches", None)


from data_manager import FeatureConfig


class DiarizationDatasetWithLength:
    """
    Wrapper for Lhotse's DiarizationDataset that adds __len__ method.

    Lhotse's DiarizationDataset doesn't have __len__ because it's designed
    to work with samplers that handle the iteration. This wrapper adds
    the missing __len__ method for PyTorch DataLoader compatibility.
    """

    def __init__(self, dataset, num_batches: int):
        self.dataset = dataset
        self._num_batches = num_batches

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return self._num_batches

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped dataset."""
        return getattr(self.dataset, name)


class LhotseSamplerWrapper:
    """
    Wrapper for Lhotse samplers to provide __len__ method for PyTorch DataLoader.

    Lhotse samplers intentionally don't have __len__() as they support streaming.
    This wrapper uses precomputed batch count for exact length reporting.

    The batch count is computed by iterating the sampler once before wrapping,
    ensuring numerical stability for schedulers and progress tracking.
    """

    def __init__(self, sampler, num_batches: Optional[int] = None):
        """
        Initialize the wrapper with a Lhotse sampler.

        Args:
            sampler: Lhotse sampler instance
            num_batches: Precomputed number of batches. If None, will be computed
                        by iterating the sampler once (only works for eager CutSets).
        """
        self.sampler = sampler
        self._num_batches = num_batches

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        """Return the precomputed number of batches."""
        if self._num_batches is None:
            raise ValueError(
                "Batch count was not precomputed. This wrapper requires knowing "
                "the exact number of batches for numerical stability. "
                "Call count_batches() before wrapping the sampler."
            )
        return self._num_batches

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped sampler."""
        return getattr(self.sampler, name)


def count_batches(sampler) -> int:
    """
    Count the exact number of batches a Lhotse sampler will yield.

    This function saves the sampler's state, iterates through all batches to count them,
    then restores the original state so the sampler can be used normally.

    Args:
        sampler: Lhotse sampler instance

    Returns:
        int: Exact number of batches the sampler will yield

    Note:
        This works by utilizing the sampler's state_dict() mechanism to restore
        its initial state after counting. For large lazy CutSets, this may take time.
    """
    # Save initial state
    initial_state = sampler.state_dict()

    # Count batches by iterating
    count = 0
    for _ in sampler:
        count += 1

    # Restore initial state
    sampler.load_state_dict(initial_state)

    return count


def create_diarization_dataloader(
    cuts: CutSet,
    batch_size: int = None,
    num_workers: int = 0,
    shuffle: bool = True,
    max_duration: Optional[float] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    feature_config: Optional[FeatureConfig] = None,
) -> DataLoader:
    """
    Create a DataLoader from Lhotse CutSet for diarization.

    Args:
        cuts: Lhotse CutSet containing audio cuts with supervisions
        batch_size: Batch size (number of cuts per batch)
        num_workers: Number of DataLoader workers (set to 0 for Lhotse)
        shuffle: Whether to shuffle the data
        max_duration: Maximum total duration per batch (for dynamic bucketing)
        drop_last: Drop last incomplete batch
        pin_memory: Pin memory for faster GPU transfer
        feature_config: Feature extraction configuration (FeatureConfig dataclass)

    Returns:
        PyTorch DataLoader

    Example:
        ```python
        from lhotse import CutSet
        from data_manager import FeatureConfig
        from training.diarization_dataloader import create_diarization_dataloader

        cuts = CutSet.from_manifests(...)
        feature_config = FeatureConfig(
            feature_type='fbank',
            num_mel_bins=80,
        )
        train_dl = create_diarization_dataloader(
            cuts=cuts,
            batch_size=32,
            shuffle=True,
            feature_config=feature_config,
        )
        ```
    """
    # Set default feature configuration
    if feature_config is None:
        feature_config = FeatureConfig()

    # DiarizationDataset requires features, so compute them if not present
    if not all(cut.has_features for cut in cuts):
        if feature_config.feature_type == "fbank":
            extractor = Fbank(
                FbankConfig(
                    sampling_rate=feature_config.sampling_rate,
                    num_mel_bins=feature_config.num_mel_bins,
                    frame_length=feature_config.frame_length,
                    frame_shift=feature_config.frame_shift,
                    dither=feature_config.dither,
                    snip_edges=feature_config.snip_edges,
                )
            )
        elif feature_config.feature_type == "mfcc":
            extractor = Mfcc(
                MfccConfig(
                    sampling_rate=feature_config.sampling_rate,
                    num_ceps=feature_config.num_mel_bins,  # MFCC uses num_ceps
                    frame_length=feature_config.frame_length,
                    frame_shift=feature_config.frame_shift,
                    dither=feature_config.dither,
                    snip_edges=feature_config.snip_edges,
                    use_energy=feature_config.use_energy,
                )
            )
        else:
            raise ValueError(f"Unsupported feature type: {feature_config.feature_type}")

        # Compute and store features
        if feature_config.storage_path:
            # Store to specified path
            cuts = cuts.compute_and_store_features(
                extractor=extractor,
                storage_path=feature_config.storage_path,
                num_jobs=feature_config.num_jobs or 1,
                mix_eagerly=feature_config.mix_eagerly,
            )
        else:
            # Store in temporary directory for in-memory-like behavior
            import tempfile

            temp_dir = tempfile.mkdtemp(prefix="lhotse_features_")
            cuts = cuts.compute_and_store_features(
                extractor=extractor,
                storage_path=temp_dir,
                num_jobs=1,
            )

    # Create sampler (will yield batches of CutSets)
    if max_duration:
        # Use BucketingSampler for variable-length sequences with duration-based batching
        # BucketingSampler is deterministic and works with eager CutSets
        sampler = BucketingSampler(
            cuts,
            sampler_type=SimpleCutSampler,
            num_buckets=10,  # Create 10 buckets by duration
            max_duration=max_duration,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    else:
        # Use simple sampler with fixed batch size
        if batch_size:
            sampler = SimpleCutSampler(
                cuts,
                max_cuts=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            sampler = SimpleCutSampler(
                cuts,
                max_duration=300.0,  # Default 5 minutes
                shuffle=shuffle,
                drop_last=drop_last,
            )

    # Precompute exact batch count for numerical stability
    num_batches = count_batches(sampler)

    # Create Lhotse's Diarization dataset from the CutSet
    dataset = DiarizationDataset(cuts)

    # Wrap sampler with precomputed batch count
    wrapped_sampler = LhotseSamplerWrapper(sampler, num_batches=num_batches)
    wrapped_sampler._num_batches = num_batches  # Store for IterableDataset

    # Create IterableDataset that bypasses PyTorch's indexing
    iterable_dataset = LhotseIterableDataset(wrapped_sampler, dataset)

    # Use IterableDataset with DataLoader
    dataloader = DataLoader(
        iterable_dataset,
        batch_size=None,  # IterableDataset handles batching
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader


def create_train_val_dataloaders(
    train_cuts: CutSet,
    val_cuts: Optional[CutSet],
    batch_size: int,
    val_batch_size: Optional[int] = None,
    num_workers: int = 0,
    max_duration: Optional[float] = None,
    label_type: str = "binary",  # Kept for compatibility
    pin_memory: bool = True,
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation DataLoaders using Lhotse's DiarizationDataset.

    Args:
        train_cuts: Training CutSet
        val_cuts: Validation CutSet (optional)
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        num_workers: Number of DataLoader workers
        max_duration: Maximum duration per batch
        label_type: Type of labels (kept for compatibility, DiarizationDataset handles this)
        pin_memory: Pin memory for GPU
        feature_config: Feature extraction configuration (FeatureConfig dataclass)

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    val_batch_size = val_batch_size or batch_size

    train_dataloader = create_diarization_dataloader(
        cuts=train_cuts,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        max_duration=max_duration,
        drop_last=True,
        pin_memory=pin_memory,
        feature_config=feature_config,
    )

    val_dataloader = None
    if val_cuts:
        val_dataloader = create_diarization_dataloader(
            cuts=val_cuts,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=False,
            max_duration=max_duration,
            drop_last=False,
            pin_memory=pin_memory,
            feature_config=feature_config,
        )

    return train_dataloader, val_dataloader
