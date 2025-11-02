#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Dataset Management and Processing

This module provides the core dataset management functionality for the TS-DIA project.
It handles dataset downloading, processing, and conversion to Lhotse CutSets for
diarization tasks, with support for 50+ speech datasets via Lhotse integration.

Key Features:
- Automatic dataset downloading and processing
- Global configuration support with dataset-specific paths
- Manifest format conversion to CutSets
- Support for custom and built-in Lhotse recipes
- Flexible parameter handling (typed + dict hybrid approach)

Main Classes:
    DatasetManager: Main class for loading and processing datasets

Key Functions:
    import_recipe: Import dataset-specific download/process functions
    select_recipe: Select appropriate recipe for a dataset
    list_available_datasets: List all supported datasets

Usage Examples:
    ```python
    from datasets import DatasetManager, parse_dataset_configs

    # Load datasets from configuration
    configs = parse_dataset_configs('configs/my_datasets.yml')
    cut_sets = DatasetManager.load_datasets(datasets=configs)

    # Process individual dataset
    from datasets import import_recipe
    process_func, download_func = import_recipe('timit')
    download_func(target_dir='./data', force_download=True)
    manifests = process_func(corpus_dir='./data/timit', output_dir='./manifests')
    ```

Command Line Usage:
    ```bash
    python -m datasets.data_manager --config configs/my_datasets.yml
    ```

Supported Datasets:
    The system supports 50+ datasets including TIMIT, LibriSpeech, VoxCeleb,
    AMI, ICSI, CHIME6, and many others via Lhotse integration.
"""
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import lhotse as lh
import torch
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    Mfcc,
    MfccConfig,
    RecordingSet,
    SupervisionSet
)
from lhotse.features.io import (
    LilcomChunkyWriter,
    LilcomFilesWriter,
    NumpyFilesWriter,
)
from lhotse.dataset import (
    DiarizationDataset,
    BucketingSampler,
    DynamicBucketingSampler,
    SimpleCutSampler,
    make_worker_init_fn,
)
from torch.utils.data import DataLoader

from data_manager import recipes
from data_manager.dataset_types import (
    FeatureConfig,
    LoadDatasetsParams,
    DataLoadingConfig,
    LabelType,
)
from data_manager.parse_args import datasets_manager_parser
from training.ego_dataset import EgoCentricDiarizationDataset


def __is_custom_recipe(dataset_name: str) -> bool:
    """
    Check if a custom recipe exists for the dataset.

    Args:
        dataset_name: Name of the dataset to check

    Returns:
        True if custom recipe exists, False otherwise
    """
    download_function_name = f"download_{dataset_name}"
    process_function_name = f"prepare_{dataset_name}"
    return (
        download_function_name in recipes.__all__
        and process_function_name in recipes.__all__
    )


def __is_implemented_dataset(dataset_name: str) -> bool:
    """
    Check if a dataset is implemented in Lhotse.

    Args:
        dataset_name: Name of the dataset to check

    Returns:
        True if both download and process functions exist in Lhotse, False otherwise
    """
    download_function_name = f"download_{dataset_name}"
    process_function_name = f"prepare_{dataset_name}"
    return (
        download_function_name in lh.recipes.__all__
        and process_function_name in lh.recipes.__all__
    )


def __is_divertion_from_standard(dataset_name: str) -> bool:
    """Checks for diffrent yet uncommon naming conventions that exist

    Args:
        dataset_name (str): the name of the dataset

    Returns:
        bool: true is the dataset has both download and prepare function.
    """
    if dataset_name == "voxceleb1" or dataset_name == "voxceleb2":
        return True
    return False


def fetch_diversion(
    dataset_name: str
) -> Tuple[
    Callable[
        ...,
        Union[
            Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]],
            Dict[str, Union[RecordingSet, SupervisionSet, CutSet]],
            Tuple[RecordingSet, SupervisionSet],
            Union[RecordingSet, SupervisionSet],
            Any,
        ],
    ],
    Optional[
        Callable[
            ...,
            Union[
                Path,
                None,
                Any,
            ],
        ]
    ],
]:
    """Fetches dataset proces function that diviate from the normal naming 

    Args:
        dataset_name (str): Dataset name

    Raises:
        ValueError: If the dataset in not one of diversent datasets

    """
    if not __is_divertion_from_standard(dataset_name):
        raise ValueError(
            f"Not in the Diversion list {dataset_name}, do not use this funciton for other than (voxceleb 1/2)")
    from lhotse.recipes.voxceleb import prepare_voxceleb
    download_function_name = f"download_{dataset_name}"
    download_function = getattr(lh.recipes, download_function_name)
    return (prepare_voxceleb, download_function)


def select_recipe(
    dataset_name: str,
) -> Tuple[
    Callable[
        ...,
        Union[
            Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]],
            Dict[str, Union[RecordingSet, SupervisionSet, CutSet]],
            Tuple[RecordingSet, SupervisionSet],
            Union[RecordingSet, SupervisionSet],
            Any,
        ],
    ],
    Optional[
        Callable[
            ...,
            Union[
                Path,
                None,
                Any,
            ],
        ]
    ],
]:
    """A Function that selects the correct recipe for the dataset

    Args:
        dataset_name: The name of the dataset

    Returns:
        Tuple of (process_function, download_function):
        - process_function: Returns manifests in various formats:
          * Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]] (nested dict - most common)
          * Dict[str, Union[RecordingSet, SupervisionSet, CutSet]] (flat dict)
          * Tuple[RecordingSet, SupervisionSet] (tuple format)
          * Union[RecordingSet, SupervisionSet] (union type)
          * Any (for functions without type annotations)
        - download_function: Returns:
          * Path (most common - path to dataset directory)
          * None (for manual download instructions)
          * Any (for functions without type annotations)

    """

    if __is_custom_recipe(dataset_name):
        download_function_name = f"download_{dataset_name}"
        download_function = getattr(recipes, download_function_name)
        process_function_name = f"prepare_{dataset_name}"
        process_function = getattr(recipes, process_function_name)
        return (process_function, download_function)
    elif __is_implemented_dataset(dataset_name):
        download_function_name = f"download_{dataset_name}"
        download_function = getattr(lh.recipes, download_function_name)
        process_function_name = f"prepare_{dataset_name}"
        process_function = getattr(lh.recipes, process_function_name)
        return (process_function, download_function)
    elif __is_divertion_from_standard(dataset_name):
        return fetch_diversion(dataset_name)
    else:
        raise ValueError(
            f"Dataset {dataset_name} is not implemented, double check the dataset name and the recipe, available datasets: {list_available_datasets()}"
        )


def list_available_datasets() -> set[str]:
    """
    List all available datasets from both Lhotse and custom recipes.

    Returns:
        Set of dataset names that can be used for downloading and processing

    Note:
        This function combines datasets from Lhotse's built-in recipes and
        any custom recipes defined in the local recipes module.
    """
    clean_names_lhotse = set(
        name.replace("download_", "").replace("prepare_", "")
        for name in lh.recipes.__all__
    )

    # Add custom recipes (empty for now but ready for expansion)
    custom_recipes = set(
        name.replace("download_", "").replace("prepare_", "")
        for name in recipes.__all__
    )

    # Always include custom recipes (even if empty) for future expansion
    all_datasets = clean_names_lhotse.union(custom_recipes)

    if not custom_recipes:
        print(
            "Note: Custom recipes package is available but empty (ready for expansion)"
        )

    return all_datasets


def import_recipe(
    dataset_name: str,
) -> Tuple[
    Callable[
        ...,
        Union[
            Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]],
            Dict[str, Union[RecordingSet, SupervisionSet, CutSet]],
            Tuple[RecordingSet, SupervisionSet],
            Union[RecordingSet, SupervisionSet],
            Any,
        ],
    ],
    Optional[
        Callable[
            ...,
            Union[
                Path,
                None,
                Any,
            ],
        ]
    ],
]:
    """A function to import the correct recipe for each dataset

    Args:
        dataset_name: The name of the dataset

    Returns:
        Tuple of (process_function, download_function):
        - process_function: Returns manifests in various formats:
          * Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]] (nested dict - most common)
          * Dict[str, Union[RecordingSet, SupervisionSet, CutSet]] (flat dict)
          * Tuple[RecordingSet, SupervisionSet] (tuple format)
          * Union[RecordingSet, SupervisionSet] (union type)
          * Any (for functions without type annotations)
        - download_function: Returns:
          * Path (most common - path to dataset directory)
          * None (for manual download instructions)
          * Any (for functions without type annotations)

    """
    if dataset_name in list_available_datasets():
        return select_recipe(dataset_name)
    else:
        raise ValueError(
            f"Dataset {dataset_name} is not implemented, double check the dataset name and the recipe, available datasets: {list_available_datasets()}"
        )


def resolve_manifest(patterns: List[Path], glob_suffix: str, manifest_dir: Path) -> Optional[Path]:
    """Resolve manifest file path from patterns or glob search.

    Args:
        patterns: List of Path objects to check for existence
        glob_suffix: Suffix pattern for glob search if no patterns exist
        manifest_dir: Directory to search for manifest files

    Returns:
        Path to the manifest file if found, None otherwise
    """
    for path in patterns:
        if path.exists():
            return path
    wildcard = next(
        (p for p in manifest_dir.glob(f"*{glob_suffix}")),
        None,
    )
    return wildcard


class DatasetManager:
    """
    Dataset manager with hybrid typed + dict parameter support.

    This class provides the main interface for loading and processing datasets.
    It supports both typed dataclass parameters and dictionary-based configurations,
    with automatic global configuration merging and manifest format conversion.

    Key Features:
    - Automatic dataset downloading and processing
    - Global configuration support with dataset-specific paths
    - Manifest format conversion to Lhotse CutSets
    - Support for 50+ speech datasets via Lhotse integration
    - Flexible parameter handling (typed + dict hybrid approach)

    Examples:
        ```python
        # Using typed parameters (recommended)
        config = DatasetConfig(
            name="librispeech",
            download_params=LibriSpeechDownloadParams(
                target_dir="./data",
                force_download=True,
                dataset_parts="mini_librispeech"
            ),
            process_params=LibriSpeechProcessParams(
                output_dir="./manifests",
                normalize_text="lower"
            )
        )

        # Using dictionary parameters (fallback)
        config = DatasetConfig(
            name="timit",
            download_params={
                "target_dir": "./data",
                "force_download": False
            },
            process_params={
                "output_dir": "./manifests",
                "num_phones": 48
            }
        )

        # Mixed approach
        config = DatasetConfig(
            name="voxceleb",
            download_params=VoxCelebDownloadParams(force_download=True),
            process_params={"output_dir": "./manifests", "num_jobs": 4}
        )

        # Load datasets
        cut_sets = DatasetManager.load_datasets(datasets=[config])
        ```
    """

    @staticmethod
    def _try_load_existing_manifests(
        output_dir: Path, dataset_name: str
    ) -> Optional[Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]]:
        """
        Try to load existing manifests from disk to skip re-preparation.

        Args:
            output_dir: Base output directory for manifests
            dataset_name: Name of the dataset

        Returns:
            Dictionary of manifests by split if they exist, None otherwise
        """

        # output_dir already includes dataset name (e.g., manifests/ava_avd)
        # So we use it directly, not append dataset_name again
        manifest_dir = output_dir
        if not manifest_dir.exists():
            return None

        # Look for standard manifest files
        # Common patterns:
        #   - recordings_train.jsonl.gz
        #   - ava_avd_recordings_train.jsonl.gz (dataset-prefixed)
        splits: set[str] = set()
        recordings_files = list(manifest_dir.glob("*recordings_*.jsonl.gz"))

        for file in recordings_files:
            # Extract split name from file
            # Examples:
            #   "recordings_train.jsonl.gz" -> "train"
            #   "ava_avd_recordings_train.jsonl.gz" -> "train"
            filename = file.stem.replace(".jsonl", "")
            # Find "recordings_" and get everything after it
            if "_recordings_" in filename:
                split_name = filename.split("_recordings_")[1]
                splits.add(split_name)
            elif filename.startswith("recordings_"):
                split_name = filename[len("recordings_"):]
                splits.add(split_name)

        if not splits:
            return None

        # Try to load manifests for each split
        manifests = {}
        for split_name in splits:
            split_manifests = {}

            # Try both with and without dataset prefix
            recordings_patterns = [
                manifest_dir / f"recordings_{split_name}.jsonl.gz",
                manifest_dir /
                f"{dataset_name}_recordings_{split_name}.jsonl.gz",
            ]
            supervisions_patterns = [
                manifest_dir / f"supervisions_{split_name}.jsonl.gz",
                manifest_dir /
                f"{dataset_name}_supervisions_{split_name}.jsonl.gz",
            ]
            cuts_patterns = [
                manifest_dir / f"cuts_{split_name}.jsonl.gz",
                manifest_dir / f"{dataset_name}_cuts_{split_name}.jsonl.gz",
            ]

            # Wildcard helpers handle recipe-specific prefixes (e.g., ami-ihm-mix_*)
            recordings_path = resolve_manifest(
                recordings_patterns, f"recordings_{split_name}.jsonl.gz", manifest_dir
            )
            supervisions_path = resolve_manifest(
                supervisions_patterns, f"supervisions_{split_name}.jsonl.gz", manifest_dir
            )
            cuts_path = resolve_manifest(
                cuts_patterns, f"cuts_{split_name}.jsonl.gz", manifest_dir
            )

            # Need at least recordings or cuts
            if not (recordings_path or cuts_path):
                continue

            # Use eager loading (not lazy) for BucketingSampler compatibility
            if recordings_path:
                split_manifests["recordings"] = RecordingSet.from_file(
                    recordings_path)

            if supervisions_path:
                split_manifests["supervisions"] = SupervisionSet.from_file(
                    supervisions_path
                )

            if cuts_path:
                split_manifests["cuts"] = CutSet.from_file(cuts_path)

            if split_manifests:
                manifests[split_name] = split_manifests

        if not manifests:
            return None

        return cast(
            Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]],
            manifests,
        )

    @staticmethod
    def _load_cached_cuts_with_features(
        storage_path: Path, split_name: str
    ) -> Optional[CutSet]:
        """
        Load cached CutSet with pre-computed features from the feature storage directory.

        Args:
            storage_path: Feature storage directory (e.g., features/ava_avd_8khz)
            split_name: Name of the split (train, dev, test, etc.)

        Returns:
            CutSet with features if cache exists, None otherwise
        """

        # Look for cached cuts in the feature storage directory
        cache_path = storage_path / f"cuts_{split_name}_with_feats.jsonl.gz"

        if cache_path.exists():
            print(
                f"  ✓ {split_name}: Loaded from cache (skip feature extraction)")
            # Use CutSet.from_file (eager) instead of load_manifest_lazy for BucketingSampler compatibility
            return CutSet.from_file(cache_path)
        print(
            f"  → {split_name}: No cache found (will extract features on first use)")
        return None

    @staticmethod
    def _compute_and_cache_features_for_split(
        cuts: CutSet,
        dataset_name: str,
        split_name: str,
        feature_cfg: FeatureConfig,
        storage_root: Path,
    ) -> CutSet:
        """
        Compute features for a single split and cache both features and the cuts-with-features manifest.

        Args:
            cuts: Source CutSet (typically without features)
            dataset_name: Name of dataset (used for per-dataset storage)
            split_name: Split name (train/val/test)
            feature_cfg: Object with feature parameters (expects attributes from global_config)
            storage_root: Root directory for features (global_config.storage_path)

        Returns:
            CutSet with features (eager) pointing to cached feature storage
        """
        # Resampling audio
        if cuts[0].sampling_rate != feature_cfg.sampling_rate:
            print(
                f"Resampling audio {cuts[0].sampling_rate} -> {feature_cfg.sampling_rate}")
            cuts = cuts.resample(feature_cfg.sampling_rate)

        # Build feature extractor from configuration
        if feature_cfg.feature_type == "fbank":
            extractor = Fbank(
                FbankConfig(
                    sampling_rate=feature_cfg.sampling_rate,
                    num_mel_bins=feature_cfg.num_mel_bins or 80,
                    frame_length=feature_cfg.frame_length,
                    frame_shift=feature_cfg.frame_shift,
                    dither=feature_cfg.dither,
                    snip_edges=feature_cfg.snip_edges,
                    round_to_power_of_two=feature_cfg.round_to_power_of_two,
                    remove_dc_offset=feature_cfg.remove_dc_offset,
                    preemph_coeff=feature_cfg.preemph_coeff,
                    window_type=feature_cfg.window_type,
                    energy_floor=feature_cfg.energy_floor,
                    raw_energy=feature_cfg.raw_energy,
                    use_fft_mag=feature_cfg.use_fft_mag,
                    low_freq=feature_cfg.low_freq,
                    high_freq=feature_cfg.high_freq,
                    torchaudio_compatible_mel_scale=feature_cfg.torchaudio_compatible_mel_scale,
                    norm_filters=feature_cfg.norm_filters,
                )
            )
        elif feature_cfg.feature_type == "mfcc":
            extractor = Mfcc(
                MfccConfig(
                    sampling_rate=feature_cfg.sampling_rate,
                    num_ceps=feature_cfg.num_ceps,
                    frame_length=feature_cfg.frame_length,
                    frame_shift=feature_cfg.frame_shift,
                    dither=feature_cfg.dither,
                    snip_edges=feature_cfg.snip_edges,
                    round_to_power_of_two=feature_cfg.round_to_power_of_two,
                    remove_dc_offset=feature_cfg.remove_dc_offset,
                    preemph_coeff=feature_cfg.preemph_coeff,
                    window_type=feature_cfg.window_type,
                    energy_floor=feature_cfg.energy_floor,
                    raw_energy=feature_cfg.raw_energy,
                    use_energy=feature_cfg.use_energy,
                    use_fft_mag=feature_cfg.use_fft_mag,
                    low_freq=feature_cfg.low_freq,
                    high_freq=feature_cfg.high_freq,
                    torchaudio_compatible_mel_scale=feature_cfg.torchaudio_compatible_mel_scale,
                    norm_filters=feature_cfg.norm_filters,
                    cepstral_lifter=feature_cfg.cepstral_lifter,
                )
            )
        else:
            raise ValueError(
                f"Unsupported feature type: {feature_cfg.feature_type}")

        # Determine parallelism and pytorch threads
        # If num_jobs <= 0, resolve to available CPU cores
        try:
            import os
            cpu_cores = os.cpu_count() or 1
        except Exception:
            cpu_cores = 1

        num_jobs = feature_cfg.num_jobs if feature_cfg.num_jobs is not None else 1
        if num_jobs <= 0:
            num_jobs = cpu_cores
        torch_threads = feature_cfg.torch_threads
        if torch_threads is None and num_jobs and num_jobs > 1:
            torch_threads = 1
        if torch_threads is not None:
            torch.set_num_threads(torch_threads)

        # Optionally window long recordings before feature extraction
        if getattr(feature_cfg, "cut_window_seconds", None):
            try:
                cut_window_seconds = getattr(
                    feature_cfg, "cut_window_seconds", None)
                if cut_window_seconds is not None:
                    window_sec = float(cut_window_seconds)
                    # Use sliding windows with no overlap for simplicity; can be made configurable
                    cuts = cuts.cut_into_windows(
                        duration=window_sec, hop=window_sec)
            except Exception as e:
                print(f"Warning: failed to window cuts: {e}")

        # Compute and store features to per-dataset directory (multiprocessing-friendly)
        dataset_storage_path = Path(storage_root) / dataset_name
        dataset_storage_path.mkdir(parents=True, exist_ok=True)

        storage_type_map: Dict[str, Union[type[LilcomChunkyWriter], type[LilcomFilesWriter], type[NumpyFilesWriter]]] = {
            "lilcom_chunky": LilcomChunkyWriter,
            "lilcom_files": LilcomFilesWriter,
            "numpy": NumpyFilesWriter,
        }
        storage_writer_cls = storage_type_map.get(
            feature_cfg.storage_type, LilcomChunkyWriter
        )

        # Convert to eager mode if lazy to avoid len() issues during feature computation
        if hasattr(cuts, "to_eager"):
            cuts = cuts.to_eager()

        compute_and_store_untyped = getattr(cuts, "compute_and_store_features")
        # Cast bound method to make the return type explicit for static checkers like Pylance.
        compute_and_store = cast(
            Callable[..., CutSet],
            compute_and_store_untyped,
        )

        cuts_with_feats: CutSet = compute_and_store(
            extractor=extractor,
            storage_path=dataset_storage_path,
            storage_type=storage_writer_cls,
            num_jobs=num_jobs,
            mix_eagerly=feature_cfg.mix_eagerly,
            progress_bar=True,
        )

        # Ensure eager CutSet for bucketing sampler compatibility
        if hasattr(cuts_with_feats, "to_eager"):
            cuts_with_feats = cuts_with_feats.to_eager()

        # Save the cuts-with-features manifest for future runs
        DatasetManager.save_cuts_with_features(
            cuts_with_feats, dataset_storage_path, split_name
        )
        # Describe the dataset
        cuts_with_feats.describe()
        return cuts_with_feats

    @staticmethod
    def save_cuts_with_features(
        cuts: CutSet, storage_path: Path, split_name: str
    ) -> None:
        """
        Save CutSet with pre-computed features to the feature storage directory.

        Args:
            cuts: CutSet with features to save
            storage_path: Feature storage directory (e.g., features/ava_avd_8khz)
            split_name: Name of the split (train, dev, test, etc.)
        """
        storage_path.mkdir(parents=True, exist_ok=True)
        cache_path = storage_path / f"cuts_{split_name}_with_feats.jsonl.gz"

        # Check if cuts actually have features before saving
        if cuts:
            # Check first few cuts to see if they have features
            sample_size = min(10, len(cuts))
            sample_cuts = [cuts[i] for i in range(sample_size)]
            if any(cut.has_features for cut in sample_cuts):
                print(
                    f"💾 Saving {split_name} cuts with features to {cache_path}")
                cuts.to_file(cache_path)
            else:
                print(
                    f"⚠️  Warning: {split_name} cuts don't have features yet, skipping save"
                )
        else:
            print(
                f"⚠️  Warning: {split_name} cuts is empty, skipping save"
            )

    @staticmethod
    def _normalize_splits(
        dataset_cut_sets: Dict[str, CutSet],
        dataset_name: str,
        val_split_ratio: float = 0.1,
    ) -> Dict[str, CutSet]:
        """
        Normalize dataset splits to unified format: train, val, test.

        Handles various split naming conventions:
        - dev → val
        - Splits train if only train exists
        - Maps test appropriately

        Args:
            dataset_cut_sets: Dictionary of CutSets by split name
            dataset_name: Name of the dataset
            val_split_ratio: Ratio to use for validation split if auto-splitting (default: 0.1)

        Returns:
            Dictionary with normalized split names (train, val, test)
        """
        normalized: Dict[str, Any] = {}

        # Map common split names
        split_mapping = {
            "dev": "val",
            "development": "val",
            "validation": "val",
            "train": "train",
            "test": "test",
            "eval": "test",
        }

        # First pass: rename splits according to mapping
        for split_name, cuts in dataset_cut_sets.items():
            normalized_name = split_mapping.get(split_name.lower(), split_name)
            normalized[normalized_name] = cuts

        # If we only have train, split it into train/val
        if "train" in normalized and "val" not in normalized:

            train_cuts = normalized["train"]

            # Calculate split point
            total_cuts = len(list(train_cuts))
            val_size = int(total_cuts * val_split_ratio)

            if val_size > 0:
                print(
                    f"Auto-splitting {dataset_name} train set: {total_cuts - val_size} train, {val_size} val"
                )
                # Split the cuts
                all_cuts_list = list(train_cuts)
                val_cuts_list = all_cuts_list[:val_size]
                train_cuts_list = all_cuts_list[val_size:]

                normalized["train"] = CutSet.from_cuts(train_cuts_list)
                normalized["val"] = CutSet.from_cuts(val_cuts_list)

        return normalized

    @staticmethod
    def _create_sampler(
        cuts: CutSet,
        data_loading: DataLoadingConfig,
    ) -> Union[BucketingSampler, DynamicBucketingSampler, SimpleCutSampler]:
        """
        Create a Lhotse sampler based on the data loading configuration.

        Args:
            cuts: The CutSet to sample from
            data_loading: Configuration object containing sampler settings

        Returns:
            An initialized Lhotse sampler instance

        Raises:
            ValueError: If an unsupported sampler type is specified
        """
        sampler_config = data_loading.sampler
        sampler_type = sampler_config.type

        if sampler_type == "bucketing":
            return BucketingSampler(
                cuts,
                max_duration=sampler_config.max_duration,
                shuffle=sampler_config.shuffle,
                drop_last=sampler_config.drop_last,
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
            return SimpleCutSampler(
                cuts,
                max_duration=sampler_config.max_duration if isinstance(
                    sampler_config.max_duration, (float, int)) and sampler_config.max_duration > 0 else None,
                shuffle=sampler_config.shuffle,
                drop_last=sampler_config.drop_last,
            )
        else:
            raise ValueError(f"Unsupported sampler type: {sampler_type}")

    @staticmethod
    def _create_dataset(
        cuts: CutSet,
        label_type: LabelType = "binary",
        data_loading: Optional[DataLoadingConfig] = None,
    ) -> Union[DiarizationDataset, Any]:
        """
        Create a diarization dataset based on the label type.

        Args:
            cuts: CutSet containing audio cuts with supervisions
            label_type: Type of labeling strategy ("ego" or "binary")
            data_loading: Data loading configuration (for frame_stack, etc.)

        Returns:
            Dataset object appropriate for the specified label type

        Raises:
            ValueError: If an unsupported label_type is provided
        """
        if label_type == "ego":
            # Import here to avoid circular dependency
            # Get frame_stack and subsampling from data_loading config if available
            frame_stack = getattr(data_loading, 'frame_stack', 1) if data_loading else 1
            subsampling = getattr(data_loading, 'subsampling', 1) if data_loading else 1
            return EgoCentricDiarizationDataset(cuts=cuts, frame_stack=frame_stack, subsampling=subsampling)
        elif label_type == "binary":
            return DiarizationDataset(cuts)
        else:
            raise ValueError(f"Unsupported label_type: {label_type}")

    @staticmethod
    def create_dataloader(
        cuts: CutSet,
        data_loading: DataLoadingConfig,
        label_type: LabelType = "binary",
        random_seed: int = 42,
    ) -> DataLoader[Any]:
        """
        Create a PyTorch DataLoader from Lhotse CutSet for diarization.

        This method creates a complete dataloader pipeline with:
        - Sampler (bucketing/dynamic_bucketing/simple)
        - Dataset (binary or ego-centric diarization)
        - DataLoader with worker initialization

        Args:
            cuts: Lhotse CutSet containing audio cuts with supervisions and precomputed features
            data_loading: Configuration for data loading (sampler, dataloader settings)
            label_type: Type of labels ("ego" or "binary")
            random_seed: Random seed for reproducibility

        Returns:
            PyTorch DataLoader configured for diarization

        Example:
            ```python
            from data_manager import DatasetManager, DataLoadingConfig
            from lhotse import CutSet

            # Load cuts with precomputed features
            cuts = CutSet.from_file("cuts_train_with_feats.jsonl.gz")

            # Create dataloader
            dataloader = DatasetManager.create_dataloader(
                cuts=cuts,
                data_loading=data_loading_config,
                label_type="binary",
                random_seed=42,
            )
            ```
        """
        # Create sampler
        sampler = DatasetManager._create_sampler(cuts=cuts, data_loading=data_loading)

        # Create dataset
        dataset = DatasetManager._create_dataset(cuts=cuts, label_type=label_type, data_loading=data_loading)

        # Create worker init function for reproducibility
        worker_init_fn = make_worker_init_fn(seed=random_seed)

        # Create DataLoader
        dataloader_cfg = data_loading.dataloader
        dataloader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=None,  # Lhotse samplers handle batching
            sampler=sampler,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=dataloader_cfg.persistent_workers if dataloader_cfg.num_workers > 0 else False,
            prefetch_factor=dataloader_cfg.prefetch_factor if dataloader_cfg.num_workers > 0 else None,
        )

        return dataloader

    @staticmethod
    def create_train_val_dataloaders(
        train_cuts: CutSet,
        val_cuts: Optional[CutSet],
        data_loading: DataLoadingConfig,
        label_type: LabelType = "binary",
        random_seed: int = 42,
    ) -> Tuple[DataLoader[Any], Optional[DataLoader[Any]]]:
        """
        Create training and validation DataLoaders for diarization.

        Args:
            train_cuts: Training CutSet with precomputed features
            val_cuts: Validation CutSet with precomputed features (optional)
            data_loading: Configuration for data loading
            label_type: Type of labels ("ego" or "binary")
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataloader, val_dataloader)

        Example:
            ```python
            from data_manager import DatasetManager

            # Load datasets
            cut_sets = DatasetManager.load_datasets(datasets=configs)
            train_cuts = cut_sets["ami"]["train"]
            val_cuts = cut_sets["ami"]["val"]

            # Create dataloaders
            train_dl, val_dl = DatasetManager.create_train_val_dataloaders(
                train_cuts=train_cuts,
                val_cuts=val_cuts,
                data_loading=data_loading_config,
                label_type="binary",
                random_seed=42,
            )
            ```
        """
        print("=" * 60)
        print("Creating training dataloader")
        print("=" * 60)

        train_dataloader = DatasetManager.create_dataloader(
            cuts=train_cuts,
            data_loading=data_loading,
            label_type=label_type,
            random_seed=random_seed,
        )

        val_dataloader = None
        if val_cuts:
            print("\n" + "=" * 60)
            print("Creating validation dataloader")
            print("=" * 60)

            # Use same config but disable shuffle for validation
            val_data_loading = DataLoadingConfig(
                strategy=data_loading.strategy,
                input_strategy=data_loading.input_strategy,
                sampler=data_loading.sampler,
                dataloader=data_loading.dataloader,
            )
            val_data_loading.sampler.shuffle = False

            val_dataloader = DatasetManager.create_dataloader(
                cuts=val_cuts,
                data_loading=val_data_loading,
                label_type=label_type,
                random_seed=random_seed,
            )

        return train_dataloader, val_dataloader

    @staticmethod
    def _download_dataset(
        dataset: Any,
        download_function: Optional[Callable[..., Union[Path, None, Any]]],
    ) -> Optional[Path]:
        """
        Download a dataset if needed.

        Args:
            dataset: Dataset configuration
            download_function: Function to download the dataset

        Returns:
            Path to downloaded corpus, or None if no download needed
        """
        if not download_function:
            return None

        dl_kwargs = dataset.get_download_kwargs()
        target_dir = dl_kwargs.get("target_dir")
        force_dl = dl_kwargs.get("force_download", False)

        # Skip download if target directory exists and force_download is False
        if target_dir:
            try:
                target_path = Path(target_dir)
                if target_path.exists() and not force_dl:
                    print(
                        f"→ Skipping download for {dataset.name}: "
                        f"target_dir {target_path} exists (force_download={force_dl})"
                    )
                    return target_path
                else:
                    return download_function(**dl_kwargs)
            except Exception:
                # Fall back to calling download function if path check fails
                return download_function(**dl_kwargs)
        else:
            # No target_dir specified, call download function
            return download_function(**dl_kwargs)

    @staticmethod
    def _prepare_process_kwargs(
        dataset: Any,
        corpus_path: Optional[Path],
        process_function: Callable[..., Any],
    ) -> Dict[str, Any]:
        """
        Prepare and validate kwargs for the process function.

        Args:
            dataset: Dataset configuration
            corpus_path: Path to corpus from download
            process_function: Function to process the dataset

        Returns:
            Filtered kwargs valid for the process function
        """
        process_kwargs = dataset.get_process_kwargs()

        # Set corpus_dir from download path if not already set
        if corpus_path and "corpus_dir" not in process_kwargs:
            process_kwargs["corpus_dir"] = corpus_path
        elif corpus_path and "corpus_dir" in process_kwargs:
            process_kwargs["corpus_dir"] = corpus_path

        # Get valid parameters for the process function
        sig = inspect.signature(process_function)
        valid_keys = set(sig.parameters.keys())

        # Map corpus_path to different parameter names if needed
        if corpus_path:
            if "audio_dir" in valid_keys:
                process_kwargs["audio_dir"] = corpus_path
            elif "data_dir" in valid_keys:
                process_kwargs["data_dir"] = corpus_path
            elif "corpus_dir" in valid_keys:
                process_kwargs["corpus_dir"] = corpus_path

        # Filter to only valid parameters
        filtered_kwargs = {
            k: v for k, v in process_kwargs.items() if k in valid_keys
        }

        # Handle parameter name mismatches
        if len(filtered_kwargs) < len(process_kwargs):
            ignored_keys = set(process_kwargs.keys()) - valid_keys

            # Try to remap corpus_dir to audio_dir or data_dir
            if "corpus_dir" in ignored_keys:
                if "audio_dir" in valid_keys:
                    filtered_kwargs["audio_dir"] = process_kwargs["corpus_dir"]
                    print("Note: 'corpus_dir' renamed to 'audio_dir' for this recipe.")
                    ignored_keys.remove("corpus_dir")
                elif "data_dir" in valid_keys:
                    filtered_kwargs["data_dir"] = process_kwargs["corpus_dir"]
                    print("Note: 'corpus_dir' renamed to 'data_dir' for this recipe.")
                    ignored_keys.remove("corpus_dir")

            if ignored_keys:
                print(
                    f"⚠️  Warning: Ignoring unsupported process parameters "
                    f"for {dataset.name}: {ignored_keys}"
                )

        return filtered_kwargs

    @staticmethod
    def _load_or_prepare_manifests(
        dataset: Any,
        process_function: Callable[..., Any],
        process_kwargs: Dict[str, Any],
    ) -> Any:
        """
        Load existing manifests or prepare new ones.

        Args:
            dataset: Dataset configuration
            process_function: Function to process the dataset
            process_kwargs: Kwargs for the process function

        Returns:
            Manifests (format varies by dataset)
        """
        output_dir = Path(process_kwargs.get("output_dir", "./manifests"))

        # Try to load existing manifests
        existing_manifests = DatasetManager._try_load_existing_manifests(
            output_dir, dataset.name
        )

        if existing_manifests:
            print(
                f"✓ Using existing manifests for {dataset.name} "
                f"(skip audio extraction & manifest creation)"
            )
            return existing_manifests
        else:
            print(
                f"→ Preparing {dataset.name} dataset "
                f"(manifests not found, will extract audio & create manifests)"
            )
            return process_function(**process_kwargs)

    @staticmethod
    def _process_features_for_dataset(
        dataset: Any,
        dataset_cut_sets: Dict[str, CutSet],
    ) -> Dict[str, CutSet]:
        """
        Load cached features or compute new ones for all splits.

        Args:
            dataset: Dataset configuration
            dataset_cut_sets: Dictionary of CutSets by split name

        Returns:
            Updated dictionary with CutSets containing features
        """
        # Get global_config
        global_config = getattr(dataset, "global_config", None)
        
        # Get data loading strategy
        data_loading = global_config.data_loading if global_config else None
        dl_strategy = (
            data_loading.strategy
            if data_loading is not None
            else "precomputed_features"
        )

        if dl_strategy != "precomputed_features":
            print(
                f"\nData loading strategy '{dl_strategy}' selected — "
                f"skipping feature precomputation."
            )
            return dataset_cut_sets

        # Process precomputed features
        print(f"\nChecking feature cache for {dataset.name}...")
        base_storage_path = global_config.storage_path if global_config else None

        if not base_storage_path:
            print("  → No storage_path configured, features will be extracted on demand")
            return dataset_cut_sets

        # Process each split
        for split_name, cuts in dataset_cut_sets.items():
            storage_path = Path(base_storage_path) / dataset.name

            # Try to load cached features
            cached_cuts = DatasetManager._load_cached_cuts_with_features(
                storage_path, split_name
            )

            if cached_cuts is not None:
                dataset_cut_sets[split_name] = cached_cuts
            else:
                # Compute and cache features
                print(
                    f"  → {split_name}: Computing features and caching to {storage_path}"
                )
                # Get feature_config from global_config
                feature_config = global_config.features if global_config else None
                if feature_config is None:
                    feature_config = FeatureConfig()

                dataset_cut_sets[split_name] = (
                    DatasetManager._compute_and_cache_features_for_split(
                        cuts,
                        dataset.name,
                        split_name,
                        feature_config,
                        Path(base_storage_path),
                    )
                )

        return dataset_cut_sets

    @staticmethod
    def _process_single_dataset(
        dataset: Any,
        process_function: Optional[Callable[..., Any]],
        download_function: Optional[Callable[..., Union[Path, None, Any]]],
        validation_split: float,
    ) -> Dict[str, CutSet]:
        """
        Process a single dataset: download, prepare manifests, extract features.

        Args:
            dataset: Dataset configuration
            process_function: Function to process the dataset
            download_function: Function to download the dataset
            validation_split: Ratio for validation split if auto-splitting

        Returns:
            Dictionary of CutSets by split name

        Raises:
            ValueError: If dataset has no download or process function
        """
        if download_function is None and process_function is None:
            raise ValueError(
                f"Dataset {dataset.name} has no download or process function"
            )

        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset.name}")
        print(f"{'=' * 60}")

        # Step 1: Download dataset
        corpus_path = DatasetManager._download_dataset(dataset, download_function)

        # Step 2: Prepare manifests
        if process_function is None:
            raise ValueError(f"Dataset {dataset.name} has no process function")

        process_kwargs = DatasetManager._prepare_process_kwargs(
            dataset, corpus_path, process_function
        )
        manifests = DatasetManager._load_or_prepare_manifests(
            dataset, process_function, process_kwargs
        )

        # Step 3: Convert manifests to CutSets
        dataset_cut_sets = DatasetManager._manifests_to_cutsets_dict(
            manifests, dataset.name
        )

        # Step 4: Normalize split names (dev → val, auto-split if needed)
        dataset_cut_sets = DatasetManager._normalize_splits(
            dataset_cut_sets, dataset.name, validation_split
        )

        # Step 5: Load or compute features
        dataset_cut_sets = DatasetManager._process_features_for_dataset(
            dataset, dataset_cut_sets
        )

        print(f"✓ {dataset.name} ready!\n")
        return dataset_cut_sets

    @staticmethod
    def load_datasets(
        datasets: List[Any],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        validation_split: float = 0.1,
        test_split: float = 0.1
    ) -> Dict[str, Dict[str, CutSet]]:
        """
        Load datasets and convert all manifest formats to CutSets for diarization tasks.

        This method handles the complete pipeline from dataset configuration to CutSet generation:
        1. Downloads datasets if needed
        2. Processes datasets to generate manifests
        3. Converts manifests to Lhotse CutSets
        4. Returns structured dictionary of CutSets organized by dataset and split

        Args:
            datasets: List of DatasetConfig objects
            batch_size: Batch size for data loading (default: 32)
            shuffle: Whether to shuffle data (default: True)
            num_workers: Number of worker processes (default: 4)
            pin_memory: Whether to pin memory (default: True)
            validation_split: Validation split ratio (default: 0.1)
            test_split: Test split ratio (default: 0.1)

        Returns:
            Dict[str, Dict[str, CutSet]]: Dictionary mapping dataset names to split dictionaries.
                Structure: {dataset_name: {split_name: CutSet}}
                Example: {"ami": {"train": CutSet(...), "dev": CutSet(...)}}

        Raises:
            ValueError: If dataset has no download or process function
            ValueError: If manifest conversion fails

        Example:
            ```python
            from datasets import DatasetManager, parse_dataset_configs

            # Single dataset
            configs = parse_dataset_configs('configs/my_datasets.yml')
            cut_sets = DatasetManager.load_datasets(datasets=configs)
            train_cuts = cut_sets["ami"]["train"]
            dev_cuts = cut_sets["ami"]["dev"]

            # Multiple datasets (cached independently)
            # First run: Downloads, extracts audio, creates manifests, extracts features
            # Second run: Loads from cache (skips all preparation)
            cut_sets = DatasetManager.load_datasets(datasets=[config1, config2, config3])

            # Each dataset cached separately
            ava_train = cut_sets["ava_avd"]["train"]  # From cache: manifests/ava_avd/
            vox_train = cut_sets["voxconverse"]["train"]  # From cache: manifests/voxconverse/
            ami_train = cut_sets["ami"]["train"]  # From cache: manifests/ami/
            ```
        """
        # Create params object for validation split ratio
        params = LoadDatasetsParams(
            datasets=datasets,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            validation_split=validation_split,
            test_split=test_split
        )

        # Import recipes for all datasets
        recipes = [
            (import_recipe(dataset.name), dataset) for dataset in params.datasets
        ]

        # Process each dataset
        all_cut_sets: Dict[str, Dict[str, CutSet]] = {}
        for (process_function, download_function), dataset in recipes:
            dataset_cut_sets = DatasetManager._process_single_dataset(
                dataset=dataset,
                process_function=process_function,
                download_function=download_function,
                validation_split=params.validation_split,
            )
            all_cut_sets[dataset.name] = dataset_cut_sets

        return all_cut_sets

    @staticmethod
    def _manifests_to_cutsets_dict(
        manifests: Any, dataset_name: str
    ) -> Dict[str, CutSet]:
        """
        Convert any manifest format to a dictionary of CutSets organized by split.

        Returns:
            Dict[str, CutSet]: Dictionary mapping split names to CutSets
                Example: {"train": CutSet(...), "dev": CutSet(...), "test": CutSet(...)}
        """
        if manifests is None:
            raise ValueError(f"Dataset {dataset_name} has no manifests")

        # If manifests is already a dict with splits
        if isinstance(manifests, dict):
            # Check if it's a nested dict (split-based)
            manifest_dict = cast(Dict[str, Any], manifests)
            if all(isinstance(v, dict) for v in manifest_dict.values()):
                # Nested dict format - has splits with manifests
                result: Dict[str, CutSet] = {}
                manifests_typed = cast(Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]], manifests)
                for split_name, split_manifests in manifests_typed.items():
                    # Convert split manifests to CutSet
                    cut_sets_list = DatasetManager._manifests_to_cutsets(
                        split_manifests, f"{dataset_name}_{split_name}"
                    )
                    # Take first CutSet (should only be one per split)
                    if cut_sets_list:
                        result[split_name] = cut_sets_list[0]
                return result

            # Check if it contains RecordingSet/SupervisionSet directly
            elif any(
                isinstance(v, (RecordingSet, SupervisionSet, CutSet))
                for v in cast(Dict[str, Any], manifests).values()
            ):
                # Single split dict - convert to CutSet and return as "train"
                cut_sets_list = DatasetManager._manifests_to_cutsets(
                    manifests, dataset_name
                )
                return {"train": cut_sets_list[0]} if cut_sets_list else {}

        # For non-dict formats (tuple, single RecordingSet, etc.), convert and return as "train"
        cut_sets_list = DatasetManager._manifests_to_cutsets(
            manifests, dataset_name)
        if not cut_sets_list:
            return {}

        # If we got multiple CutSets from a non-dict format, name them sequentially
        if len(cut_sets_list) == 1:
            return {"train": cut_sets_list[0]}
        else:
            # Multiple cuts - try to infer split names or use sequential naming
            split_names = ["train", "dev", "test"]
            return {
                split_names[i] if i < len(split_names) else f"split_{i}": cutset
                for i, cutset in enumerate(cut_sets_list)
            }

    @staticmethod
    def _manifests_to_cutsets(manifests: Any, dataset_name: str) -> List[CutSet]:
        """Convert any manifest format to CutSet(s) for diarization tasks"""
        cut_sets: List[CutSet] = []

        if manifests is None:
            raise ValueError(f"Dataset {dataset_name} has no manifests")

        # Handle Tuple format: (RecordingSet, SupervisionSet)
        if isinstance(manifests, tuple):
            manifests_tuple = cast(Tuple[Any, ...], manifests)
            if len(manifests_tuple) == 2:
                recording_set, supervision_set = cast(
                    Tuple[Optional[RecordingSet], Optional[SupervisionSet]],
                    manifests_tuple,
                )
                if recording_set is not None and supervision_set is not None:
                    cut_sets.append(
                        CutSet.from_manifests(
                            recordings=recording_set,
                            supervisions=supervision_set,
                        )
                    )

        # Handle Union format: RecordingSet or SupervisionSet
        elif isinstance(manifests, (RecordingSet, SupervisionSet)):
            if isinstance(manifests, RecordingSet):
                # If only RecordingSet, create empty supervisions
                cut_sets.append(
                    CutSet.from_manifests(
                        recordings=manifests,
                        supervisions=None,
                    )
                )
            else:
                # If only SupervisionSet, we need recordings - this is unusual
                raise ValueError(
                    f"Dataset {dataset_name} returned only SupervisionSet without recordings"
                )

        # Handle Dict formats
        elif isinstance(manifests, dict):
            manifest_dict = cast(Dict[str, Any], manifests)
            # Check if it's a nested dict (split-based): Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]
            if all(isinstance(v, dict) for v in manifest_dict.values()):
                # Nested dict format - iterate through splits
                nested_manifests = cast(Dict[str, Dict[str, Any]], manifest_dict)
                for split_name, split_manifests in nested_manifests.items():
                    split_cut_sets = DatasetManager._manifests_to_cutsets(
                        split_manifests, f"{dataset_name}_{split_name}"
                    )
                    cut_sets.extend(split_cut_sets)

            # Check if it's a flat dict: Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]
            elif any(
                isinstance(v, (RecordingSet, SupervisionSet, CutSet))
                for v in manifest_dict.values()
            ):
                recording_candidate = manifest_dict.get("recordings")
                if recording_candidate is None:
                    recording_candidate = manifest_dict.get("recording")
                recording_set = cast(Optional[RecordingSet], recording_candidate)

                supervision_candidate = manifest_dict.get("supervisions")
                if supervision_candidate is None:
                    supervision_candidate = manifest_dict.get("supervision")
                supervision_set = cast(
                    Optional[SupervisionSet], supervision_candidate
                )

                # Check if we have a CutSet directly
                cuts_value = manifest_dict.get("cuts")
                if isinstance(cuts_value, CutSet):
                    cut_sets.append(cuts_value)
                elif recording_set is not None and supervision_set is not None:
                    cut_sets.append(
                        CutSet.from_manifests(
                            recordings=recording_set,
                            supervisions=supervision_set,
                        )
                    )
                elif recording_set is not None:
                    # Only recordings available
                    cut_sets.append(
                        CutSet.from_manifests(
                            recordings=recording_set,
                            supervisions=None,
                        )
                    )
                else:
                    raise ValueError(
                        f"Dataset {dataset_name} dict format has no recognizable manifests"
                    )

            else:
                raise ValueError(
                    f"Dataset {dataset_name} dict format is not recognized"
                )

        # Handle Any type (functions without type annotations)
        else:
            # Try to extract manifests from unknown format
            # First check if it's not a tuple (which we already handled above)
            if not isinstance(manifests, tuple) and hasattr(manifests, "recordings") and hasattr(manifests, "supervisions"):
                cut_sets.append(
                    CutSet.from_manifests(
                        recordings=manifests.recordings,
                        supervisions=manifests.supervisions,
                    )
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} returned unrecognized manifest format"
                )

        if not cut_sets:
            raise ValueError(
                f"Dataset {dataset_name} could not be converted to CutSet(s)"
            )

        return cut_sets


if __name__ == "__main__":
    args, dataset_configs = datasets_manager_parser()

    # Create LoadDatasetsParams with the parsed dataset configurations
    load_params = LoadDatasetsParams(
        datasets=dataset_configs,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        validation_split=0.1,
        test_split=0.1,
    )

    cut_sets = DatasetManager.load_datasets(**vars(load_params))

    print(f"Loaded {len(cut_sets)} datasets")
    for dataset_name, splits in cut_sets.items():
        print(f"\nDataset: {dataset_name}")
        for split_name, cut_set in splits.items():
            print(f"  Split: {split_name}")
            cut_set.describe()
