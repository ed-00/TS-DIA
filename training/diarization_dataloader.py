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

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import torch
from lhotse import CutSet
from lhotse.dataset import BucketingSampler, DiarizationDataset, SimpleCutSampler
from lhotse.dataset.collation import collate_matrices
from lhotse.dataset.dataloading import make_worker_init_fn
from lhotse.dataset.input_strategies import (
    OnTheFlyFeatures,
)
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from data_manager import FeatureConfig
from data_manager.dataset_types import DataLoadingConfig


def create_diarization_dataloader(
    cuts: CutSet,
    batch_size: int = None,
    num_workers: int = 0,
    shuffle: bool = True,
    max_duration: Optional[float] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    feature_config: Optional[FeatureConfig] = None,
    dataset_name: Optional[str] = None,
    split_name: str = "train",
    base_storage_path: Optional[str] = None,
    min_speaker_dim: Optional[int] = None,
    data_loading: Optional[DataLoadingConfig] = None,
    random_seed: Optional[int] = 42,
) -> Tuple[DataLoader, CutSet]:
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
        dataset_name: Name of the dataset for caching (optional)
        split_name: Name of the split for caching (train/val/test)
        base_storage_path: Base storage path for caching (optional)

    Returns:
        Tuple of (PyTorch DataLoader, CutSet with features)

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
    # Expect features to be precomputed and cached in DatasetManager.load_datasets()
    # Window overly long recordings before sampling to match feature cache layout
    if feature_config and getattr(feature_config, "cut_window_seconds", None):
        try:
            window_sec = float(feature_config.cut_window_seconds)
            if window_sec > 0:
                cuts = cuts.cut_into_windows(duration=window_sec, hop=window_sec)
        except Exception:
            pass

    # Build InputStrategy from YAML config if provided
    input_strategy = None
    dataloader_cfg_workers = num_workers
    if data_loading is not None:
        # Resolve executor type
        exec_cls = (
            ThreadPoolExecutor
            if data_loading.input_strategy.executor_type == "thread"
            else ProcessPoolExecutor
        )

        if data_loading.strategy == "on_the_fly_features":
            # Build feature extractor
            if feature_config is None:
                raise ValueError(
                    "On-the-fly features strategy requires feature_config to be provided."
                )
            from lhotse import Fbank, FbankConfig, Mfcc, MfccConfig

            if feature_config.feature_type == "fbank":
                extractor = Fbank(
                    FbankConfig(
                        sampling_rate=feature_config.sampling_rate,
                        num_mel_bins=feature_config.num_mel_bins or 80,
                        frame_length=feature_config.frame_length,
                        frame_shift=feature_config.frame_shift,
                        dither=feature_config.dither,
                        snip_edges=feature_config.snip_edges,
                        round_to_power_of_two=feature_config.round_to_power_of_two,
                        remove_dc_offset=feature_config.remove_dc_offset,
                        preemph_coeff=feature_config.preemph_coeff,
                        window_type=feature_config.window_type,
                        energy_floor=feature_config.energy_floor,
                        raw_energy=feature_config.raw_energy,
                        use_fft_mag=feature_config.use_fft_mag,
                        low_freq=feature_config.low_freq,
                        high_freq=feature_config.high_freq,
                        torchaudio_compatible_mel_scale=feature_config.torchaudio_compatible_mel_scale,
                        norm_filters=feature_config.norm_filters,
                    )
                )
            elif feature_config.feature_type == "mfcc":
                extractor = Mfcc(
                    MfccConfig(
                        sampling_rate=feature_config.sampling_rate,
                        num_ceps=feature_config.num_ceps,
                        frame_length=feature_config.frame_length,
                        frame_shift=feature_config.frame_shift,
                        dither=feature_config.dither,
                        snip_edges=feature_config.snip_edges,
                        round_to_power_of_two=feature_config.round_to_power_of_two,
                        remove_dc_offset=feature_config.remove_dc_offset,
                        preemph_coeff=feature_config.preemph_coeff,
                        window_type=feature_config.window_type,
                        energy_floor=feature_config.energy_floor,
                        raw_energy=feature_config.raw_energy,
                        use_energy=feature_config.use_energy,
                        use_fft_mag=feature_config.use_fft_mag,
                        low_freq=feature_config.low_freq,
                        high_freq=feature_config.high_freq,
                        torchaudio_compatible_mel_scale=feature_config.torchaudio_compatible_mel_scale,
                        norm_filters=feature_config.norm_filters,
                        cepstral_lifter=feature_config.cepstral_lifter,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported feature type for on-the-fly strategy: {feature_config.feature_type}"
                )

            input_strategy = OnTheFlyFeatures(
                extractor=extractor,
                num_workers=data_loading.input_strategy.num_workers,
                use_batch_extract=data_loading.input_strategy.use_batch_extract,
                fault_tolerant=data_loading.input_strategy.fault_tolerant,
                return_audio=data_loading.input_strategy.return_audio,
                executor_type=exec_cls,
            )
        elif data_loading.strategy == "audio_samples":
            # Not supported in diarization path (labels align to frames).
            raise ValueError(
                "audio_samples strategy is not supported in diarization pipeline."
            )

        # If InputStrategy has its own workers, ensure DataLoader workers are disabled
        if input_strategy is not None:
            if (
                data_loading.input_strategy.num_workers
                and data_loading.input_strategy.num_workers > 0
            ):
                dataloader_cfg_workers = 0

    # Choose Lhotse sampler (consider lazy vs eager and YAML config)
    if data_loading is not None and data_loading.sampler.type == "dynamic_bucketing":
        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=data_loading.sampler.max_duration or max_duration,
            num_buckets=data_loading.sampler.num_buckets,
            shuffle=data_loading.sampler.shuffle if shuffle is None else shuffle,
            drop_last=data_loading.sampler.drop_last
            if drop_last is None
            else drop_last,
            seed=random_seed or 0,
        )
    else:
        # If max_duration provided (via arg or YAML), prefer BucketingSampler
        eff_max_duration = (
            data_loading.sampler.max_duration if data_loading else None
        ) or max_duration
        if eff_max_duration:
            # BucketingSampler doesn't support lazy CutSets; auto-switch to DynamicBucketingSampler
            if getattr(cuts, "is_lazy", False):
                sampler = DynamicBucketingSampler(
                    cuts,
                    max_duration=eff_max_duration,
                    num_buckets=(
                        data_loading.sampler.num_buckets if data_loading else 10
                    ),
                    shuffle=(data_loading.sampler.shuffle if data_loading else shuffle),
                    drop_last=(
                        data_loading.sampler.drop_last if data_loading else drop_last
                    ),
                    seed=random_seed or 0,
                )
            else:
                sampler = BucketingSampler(
                    cuts,
                    sampler_type=SimpleCutSampler,
                    num_buckets=(
                        data_loading.sampler.num_buckets if data_loading else 10
                    ),
                    max_duration=eff_max_duration,
                    shuffle=(data_loading.sampler.shuffle if data_loading else shuffle),
                    drop_last=(
                        data_loading.sampler.drop_last if data_loading else drop_last
                    ),
                )
        else:
            # SimpleCutSampler by number of cuts
            eff_batch_size = batch_size
            if eff_batch_size is None and data_loading is not None:
                # No special default from YAML; keep None to fall back to max_duration in SimpleCutSampler
                pass
            if eff_batch_size:
                sampler = SimpleCutSampler(
                    cuts,
                    max_cuts=eff_batch_size,
                    shuffle=(data_loading.sampler.shuffle if data_loading else shuffle),
                    drop_last=(
                        data_loading.sampler.drop_last if data_loading else drop_last
                    ),
                )
            else:
                sampler = SimpleCutSampler(
                    cuts,
                    max_duration=300.0,
                    shuffle=(data_loading.sampler.shuffle if data_loading else shuffle),
                    drop_last=(
                        data_loading.sampler.drop_last if data_loading else drop_last
                    ),
                )

    # Create dataset depending on strategy
    if (
        data_loading is not None
        and data_loading.strategy == "on_the_fly_features"
        and input_strategy is not None
    ):
        frame_shift = feature_config.frame_shift if feature_config else 0.01

        class _DiarizationOnTheFlyDataset(torch.utils.data.Dataset):
            def __init__(self, cuts: CutSet, min_speaker_dim: Optional[int] = None):
                self.cuts = cuts
                self.min_speaker_dim = min_speaker_dim

            def __getitem__(self, cuts_batch: CutSet):
                feats, feat_lens, *rest = input_strategy(cuts_batch)

                # Build per-cut speaker activity matrices in frame domain using feature_config.frame_shift
                masks = []
                for idx, cut in enumerate(cuts_batch):
                    num_frames = int(feat_lens[idx].item())
                    speakers = sorted(set(s.speaker for s in cut.supervisions))

                    # Handle speaker indexing
                    speaker_to_idx = {spk: i for i, spk in enumerate(speakers)}
                    num_speakers = len(speakers)

                    # Apply min_speaker_dim regardless of whether speakers exist
                    if self.min_speaker_dim is not None:
                        num_speakers = max(num_speakers, int(self.min_speaker_dim))

                    # Create mask with proper dimensions
                    mask = torch.zeros((num_speakers, num_frames), dtype=torch.long)

                    # Fill in speaker activity (only if speakers exist)
                    if speakers:
                        for sup in cut.supervisions:
                            spk_idx = speaker_to_idx[sup.speaker]
                            st = int(round(max(0.0, sup.start) / frame_shift))
                            et = int(round(min(cut.duration, sup.end) / frame_shift))
                            st = max(0, min(st, num_frames))
                            et = max(0, min(et, num_frames))
                            if et > st:
                                mask[spk_idx, st:et] = 1

                    # Transpose to [num_frames, num_speakers] for collate_matrices
                    # (collate_matrices pads along the first dimension)
                    masks.append(mask.T)

                # Collate masks: input is list of [num_frames, num_speakers], output is [batch, max_frames, num_speakers]
                speaker_activity = collate_matrices(
                    masks, padding_value=CrossEntropyLoss().ignore_index
                )

                # Transpose to [batch, num_speakers, max_frames] to match expected format
                speaker_activity = speaker_activity.transpose(1, 2)

                return {
                    "features": feats,
                    "features_lens": feat_lens,
                    "speaker_activity": speaker_activity,
                }

        dataset = _DiarizationOnTheFlyDataset(cuts, min_speaker_dim=min_speaker_dim)
    else:
        # Default precomputed-features path
        dataset = DiarizationDataset(cuts, min_speaker_dim=min_speaker_dim)

    # Use native Lhotse dataset + sampler with DataLoader
    worker_init_fn = None
    if dataloader_cfg_workers and dataloader_cfg_workers > 0:
        # Seed workers distinctly for randomized ops; rank/world_size inferred when possible
        worker_init_fn = make_worker_init_fn(seed=random_seed or 0)

    # Use Lhotse sampler with batch_size=None (per Lhotse docs)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=dataloader_cfg_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    return (
        dataloader,
        cuts,
    )  # Return both dataloader and cuts (with features if computed)


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
    dataset_name: Optional[str] = None,
    base_storage_path: Optional[str] = None,
    drop_last: bool = False,
    min_speaker_dim: Optional[int] = None,
    data_loading: Optional[DataLoadingConfig] = None,
    random_seed: Optional[int] = 42,
) -> Tuple[DataLoader, Optional[DataLoader], CutSet, Optional[CutSet]]:
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
        dataset_name: Name of the dataset for caching (optional)
        base_storage_path: Base storage path for caching (optional)

    Returns:
        Tuple of (train_dataloader, val_dataloader, train_cuts_with_feats, val_cuts_with_feats)
    """
    val_batch_size = val_batch_size or batch_size

    # Prefer cached windowed cuts-with-features when available
    resolved_train_cuts = train_cuts
    if dataset_name and base_storage_path:
        train_cache = (
            Path(base_storage_path) / dataset_name / "cuts_train_with_feats.jsonl.gz"
        )
        if train_cache.exists():
            try:
                resolved_train_cuts = CutSet.from_file(train_cache)
            except Exception:
                pass

    train_dataloader, train_cuts_with_feats = create_diarization_dataloader(
        cuts=resolved_train_cuts,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        max_duration=max_duration,
        drop_last=True,
        pin_memory=pin_memory,
        feature_config=feature_config,
        dataset_name=dataset_name,
        split_name="train",
        base_storage_path=base_storage_path,
        min_speaker_dim=min_speaker_dim,
        data_loading=data_loading,
        random_seed=random_seed,
    )

    val_dataloader = None
    val_cuts_with_feats = None
    if val_cuts:
        resolved_val_cuts = val_cuts
        if dataset_name and base_storage_path:
            val_cache = (
                Path(base_storage_path) / dataset_name / "cuts_val_with_feats.jsonl.gz"
            )
            if val_cache.exists():
                try:
                    resolved_val_cuts = CutSet.from_file(val_cache)
                except Exception:
                    pass

        val_dataloader, val_cuts_with_feats = create_diarization_dataloader(
            cuts=resolved_val_cuts,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=False,
            max_duration=max_duration,
            drop_last=drop_last,
            pin_memory=pin_memory,
            feature_config=feature_config,
            dataset_name=dataset_name,
            split_name="val",
            base_storage_path=base_storage_path,
            min_speaker_dim=min_speaker_dim,
            data_loading=data_loading,
            random_seed=random_seed,
        )

    return train_dataloader, val_dataloader, train_cuts_with_feats, val_cuts_with_feats
