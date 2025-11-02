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
from typing import Optional, Tuple, Union
from pathlib import Path

from data_manager import FeatureConfig, LabelType
from data_manager.dataset_types import DataLoadingConfig
from training.ego_dataset import EgoCentricDiarizationDataset

from lhotse.features import FeatureExtractor

from lhotse import (
    FbankConfig,
    MfccConfig,
    CutSet,
    Fbank,
    Mfcc,
)

from lhotse.dataset import (
    DynamicBucketingSampler,
    PrecomputedFeatures,
    DiarizationDataset,
    make_worker_init_fn,
    BucketingSampler,
    SimpleCutSampler,
    OnTheFlyFeatures,
)


from torch.utils.data import Dataset, DataLoader
from typing import Any


def extractor_generator(feature_config: FeatureConfig) -> FeatureExtractor:
    """Generate a feature extractor based on the provided configuration.

    This function creates and returns an appropriate feature extractor (Fbank or MFCC)
    based on the feature type specified in the configuration. It supports two types
    of audio feature extraction: filter bank (fbank) and Mel-frequency cepstral
    coefficients (MFCC).

        feature_config (FeatureConfig): Configuration object containing feature
            extraction parameters including feature type, sampling rate, frame
            parameters, and various processing options.

        ValueError: If the feature_config.feature_type is not "fbank" or "mfcc".

        FeatureExtractor: An initialized feature extractor object (either Fbank
            or Mfcc) configured with the specified parameters.

    Example:
        >>> config = FeatureConfig(feature_type="fbank", sampling_rate=16000)
        >>> extractor = extractor_generator(config)
        >>> features = extractor.extract(audio_data)
    """
    if feature_config.feature_type == "fbank":
        print("Extractiong fbanks...")
        extractor = Fbank(
            FbankConfig(
                sampling_rate=feature_config.sampling_rate,
                num_mel_bins=feature_config.num_mel_bins,
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
        print("Extracting mfcc...")
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
            f"Unsupported feature type for strategy: {feature_config.feature_type}"
        )
    return extractor


def input_strategy_generator(
    extractor: FeatureExtractor,
    data_loading: DataLoadingConfig,
    exec_cls: Union[type[ThreadPoolExecutor], type[ProcessPoolExecutor]]

) -> Union[PrecomputedFeatures, OnTheFlyFeatures]:
    """Generate an input strategy for diarization data loading based on configuration.

    This function creates and returns an appropriate input strategy object based on the 
    specified data loading strategy. It supports on-the-fly feature extraction and 
    precomputed features strategies for diarization pipelines.
    args:
        extractor (Union[Mfcc, Fbank]): Feature extractor instance (MFCC or Fbank) 
            used for audio feature extraction.
        data_loading (DataLoadingConfig): Configuration object containing data loading 
            parameters including strategy type and input strategy settings.
        exec_cls (Union[ThreadPoolExecutor, ProcessPoolExecutor]): Executor class type 
            for parallel processing during feature extraction.

        ValueError: If the specified strategy is "audio_samples" which is not supported 
            in diarization pipelines due to frame-label alignment requirements.

        Union[BatchIO, PrecomputedFeatures, AudioSamples, OnTheFlyFeatures]: 
            An input strategy object configured according to the specified strategy:
            - OnTheFlyFeatures: For real-time feature extraction during data loading
            - PrecomputedFeatures: For loading pre-extracted features from storage

    Note:
        The "audio_samples" strategy is intentionally not supported as diarization 
        requires frame-level alignment with speaker labels.
    """
    if data_loading.strategy == "on_the_fly_features":
        print("On the fly features stategy is choosen...")
        # Build feature extractor
        input_strategy = OnTheFlyFeatures(
            extractor=extractor,
            num_workers=data_loading.input_strategy.num_workers,
            use_batch_extract=data_loading.input_strategy.use_batch_extract,
            fault_tolerant=data_loading.input_strategy.fault_tolerant,
            return_audio=data_loading.input_strategy.return_audio,
            executor_type=exec_cls
        )
    elif data_loading.strategy == "precomputed_features":
        input_strategy = PrecomputedFeatures(
            num_workers=data_loading.input_strategy.num_workers,
            executor_type=exec_cls,
        )
    else:
        # Not supported in diarization path (labels align to frames).
        raise ValueError(
            "audio_samples strategy is not supported in diarization pipeline."
        )
    return input_strategy


def sampler_generator(
    cuts: CutSet,
    data_loading: DataLoadingConfig,
) -> Union[BucketingSampler, DynamicBucketingSampler, SimpleCutSampler]:
    """
    Generate a Lhotse sampler based on the provided configuration.

    Args:
        cuts: The CutSet to sample from.
        data_loading: Configuration object containing sampler settings.

    Returns:
        An initialized Lhotse sampler instance.

    Raises:
        ValueError: If an unsupported sampler type is specified in the config.
    """
    # Access the sampler-specific configuration
    sampler_config = data_loading.sampler
    sampler_type = sampler_config.type

    print(f"Initializing sampler of type: {sampler_type}")

    if sampler_type == "bucketing":
        return BucketingSampler(
            cuts,
            # max_duration is the main constraint for batch size
            max_duration=sampler_config.max_duration,
            shuffle=sampler_config.shuffle,
            drop_last=sampler_config.drop_last,
            # num_buckets controls the grouping of similar-duration cuts
            num_buckets=sampler_config.num_buckets,
        )
    elif sampler_type == "dynamic_bucketing":
        return DynamicBucketingSampler(
            cuts,
            max_duration=sampler_config.max_duration,
            shuffle=sampler_config.shuffle,
            drop_last=sampler_config.drop_last,
            num_buckets=sampler_config.num_buckets,
        )
    elif sampler_type == "simple":
        # SimpleCutSampler doesn't use buckets
        return SimpleCutSampler(
            cuts,
            max_duration=sampler_config.max_duration if isinstance(
                sampler_config.max_duration, float) else 0,
            shuffle=sampler_config.shuffle,
            drop_last=sampler_config.drop_last,
        )
    else:
        # Handle any unsupported sampler types
        raise ValueError(f"Unsupported sampler type in config: {sampler_type}")


def dataset_generator(cuts: CutSet, lable_type: LabelType = "ego") -> Union[DiarizationDataset, EgoCentricDiarizationDataset]:
    """
    Creates and returns a dataset object based on the specified label type for diarization tasks.

    This function serves as a factory method to instantiate different types of diarization 
    datasets depending on the labeling approach required for training or evaluation.
    Ags:
        cuts (CutSet): A collection of audio cuts containing the audio data and metadata
                      required for diarization processing.
        lable_type (LabelType, optional): The type of labeling strategy to use for the dataset.
                                        - "ego": Creates an ego-centric diarization dataset
                                        - "binary": Creates a binary diarization dataset
                                        Defaults to "ego".

        Dataset: A dataset object appropriate for the specified label type. Returns either
                an EgoCentricDiarizationDataset for ego-centric labeling or a 
                DiarizationDataset for binary labeling.

    Raises:
        ValueError: If an unsupported lable_type is provided (neither "ego" nor "binary").
    """
    if lable_type == "ego":
        return EgoCentricDiarizationDataset(
            cuts=cuts,
        )
    else:
        return DiarizationDataset(
            cuts
        )


def create_diarization_dataloader(
    cuts: CutSet,
    batch_size: Optional[int] = None,
    label_type: LabelType = "ego",
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
) -> Tuple[DataLoader[Any], CutSet]:
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
    if not data_loading:
        raise ValueError("No data_loading config were passed")

    if not feature_config:
        raise ValueError(
            "On-the-fly features strategy requires feature_config to be provided."
        )

    print("="*60)
    print("Creating dataloader")
    print("="*60)

    # Build InputStrategy from YAML config if provided
    dataloader_cfg_workers = num_workers

    # Resolve executor type
    exec_cls = (
        ThreadPoolExecutor
        if data_loading.input_strategy.executor_type == "thread"
        else ProcessPoolExecutor
    )
    # Generate extractor
    extractor = extractor_generator(feature_config)

    # Generate input strategy
    feature_strategy = input_strategy_generator(
        extractor=extractor,
        data_loading=data_loading,
        exec_cls=exec_cls
    )

    cuts = cuts.compute_and_store_features(
        extractor,
    )
    # Generate sampler
    sampler = sampler_generator(
        cuts=cuts,
        data_loading=data_loading
    )

    # Generate worker init funciton
    worker_init_fn = make_worker_init_fn(
        seed=random_seed
    )

    # extract features

    # Generate dataset
    dataset = dataset_generator(
        cuts=cuts, lable_type=label_type
    )
    dataloader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=dataloader_cfg_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
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
            Path(base_storage_path) / dataset_name /
            "cuts_train_with_feats.jsonl.gz"
        )
        if train_cache.exists():
            resolved_train_cuts = CutSet.from_file(train_cache)

    resolved_train_cuts.describe()

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
                Path(base_storage_path) / dataset_name /
                "cuts_val_with_feats.jsonl.gz"
            )
            if val_cache.exists():
                resolved_val_cuts = CutSet.from_file(val_cache)

        resolved_val_cuts.describe()
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
