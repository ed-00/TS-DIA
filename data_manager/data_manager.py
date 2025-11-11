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
from lhotse import (
    CutSet,
    KaldifeatFbankConfig,
    KaldifeatFbank,
    KaldifeatMfcc,
    KaldifeatMfccConfig,
    RecordingSet,
    SupervisionSet
)

from lhotse.features.kaldifeat import (
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
)
from lhotse.features.io import (
    LilcomChunkyWriter,
    LilcomFilesWriter,
    NumpyFilesWriter,
)
from lhotse.dataset import (
    DiarizationDataset,
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
import torch


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
        output_dir: Path, dataset_name: str, storage_path: Optional[Path] = None
    ) -> Optional[Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]]:
        """
        Try to load existing manifests from disk to skip re-preparation.
        Prioritizes cached CutSets with features if available.

        Args:
            output_dir: Base output directory for manifests
            dataset_name: Name of the dataset
            storage_path: Optional storage path for cached features

        Returns:
            Dictionary of manifests by split if they exist, None otherwise
        """

        # First, try to load cached CutSets with features if storage_path is provided
        if storage_path is not None:
            print("="*60)
            print(
                f"Checking for cached CutSets with features in {storage_path}")
            print("="*60)

            # Look for dataset split directories like a
            cached_manifests = {}
            storage_root = Path(storage_path)
            candidate_dirs: List[Path] = []

            dataset_storage_path = storage_root / dataset_name
            if dataset_storage_path.exists():
                candidate_dirs.append(dataset_storage_path)

            if storage_root.exists():
                candidate_dirs.append(storage_root)

            seen_files: set[Path] = set()

            for candidate_dir in candidate_dirs:
                for manifest_file in candidate_dir.glob("**/*_with_feats.jsonl.gz"):
                    if manifest_file in seen_files:
                        continue

                    # Ensure the file is associated with the dataset
                    if dataset_name not in manifest_file.parts and not manifest_file.name.startswith(f"{dataset_name}_"):
                        continue

                    filename = manifest_file.name.replace(".jsonl.gz", "")
                    split_candidate = filename

                    if split_candidate.startswith(f"{dataset_name}_"):
                        split_candidate = split_candidate[len(
                            dataset_name) + 1:]

                    if split_candidate.startswith("cuts_"):
                        split_candidate = split_candidate[len("cuts_"):]

                    if split_candidate.endswith("_with_feats"):
                        split_name = split_candidate[: -len("_with_feats")]
                    else:
                        continue

                    if not split_name:
                        continue

                    try:
                        # Use lazy loading to avoid loading large datasets into memory
                        cuts = CutSet.from_jsonl_lazy(manifest_file)
                    except Exception as exc:
                        print(
                            f"  → Failed to load cached CutSet from {manifest_file}: {exc}"
                        )
                        continue

                    seen_files.add(manifest_file)
                    cached_entry = cached_manifests.setdefault(split_name, {})
                    cached_entry["cuts"] = cuts
                    print(
                        f"  ✓ {split_name}: Loaded cached CutSet with features (lazy) from {manifest_file}"
                    )

            if cached_manifests:
                print(
                    f"✓ Using cached CutSets with features for {dataset_name} (skip manifest loading)")
                print("="*60)
                return cast(
                    Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]],
                    cached_manifests,
                )

        # Fall back to loading raw manifests if no cached features found
        # output_dir already includes dataset name (e.g., manifests/ava_avd)
        # So we use it directly, not append dataset_name again
        manifest_dir = output_dir
        if not manifest_dir.exists():
            print("="*60)
            print(f"Manifest dir does not exist!")
            print("="*60)
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

            print("="*60)
            print("Resolved Manifests")
            # Need at least recordings or cuts
            if not (recordings_path or cuts_path):
                continue

            # Use lazy loading to avoid memory issues with large datasets
            if recordings_path:
                split_manifests["recordings"] = RecordingSet.from_jsonl_lazy(
                    recordings_path)
                print(f"- Loaded recordings (lazy) from: {recordings_path}")

            if supervisions_path:
                split_manifests["supervisions"] = SupervisionSet.from_jsonl_lazy(
                    supervisions_path
                )
                print(f"- Loaded supervision (lazy) from: {supervisions_path}")

            if cuts_path:
                split_manifests["cuts"] = CutSet.from_jsonl_lazy(cuts_path)
                print(f"- Loaded cutset (lazy): {cuts_path}")

            if split_manifests:
                manifests[split_name] = split_manifests
            print("="*60)
            print("\n")

        if not manifests:
            return None

        return cast(
            Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]],
            manifests,
        )

    @staticmethod
    def _load_cached_cuts_with_features(
        storage_path: Path, split_name: str, dataset_name: str
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
        cache_path = storage_path / \
            f"{dataset_name}_{split_name}_with_feats.jsonl.gz"

        if cache_path.exists():
            print(
                f"  ✓ {split_name}: Loaded from cache (lazy loading)")
            # Use lazy loading to avoid memory issues with large datasets
            return CutSet.from_jsonl_lazy(cache_path)
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

        This method uses memory-efficient lazy loading to handle large datasets:
        - Features are computed and written to disk incrementally in batches
        - The returned CutSet uses lazy loading (only loads metadata, not feature arrays)
        - Actual features are loaded on-demand during training/inference

        This prevents memory exhaustion when processing large datasets.

        Args:
            cuts: Source CutSet (typically without features)
            dataset_name: Name of dataset (used for per-dataset storage)
            split_name: Split name (train/val/test)
            feature_cfg: Object with feature parameters (expects attributes from global_config)
            storage_root: Root directory for features (global_config.storage_path)

        Returns:
            CutSet with features (lazy) pointing to cached feature storage
        """
        # Check and resample if needed - this must be done before feature extraction
        # Note: resample() on lazy CutSets still iterates lazily
        first_cut = next(iter(cuts))
        if first_cut.sampling_rate != feature_cfg.sampling_rate:
            print(
                f"Resampling audio {first_cut.sampling_rate} -> {feature_cfg.sampling_rate}")
            cuts = cuts.resample(feature_cfg.sampling_rate)

        # Build feature extractor from configuration
        if feature_cfg.feature_type == "fbank":
            extractor = KaldifeatFbank(
                KaldifeatFbankConfig(
                    frame_opts=KaldifeatFrameOptions(
                        sampling_rate=feature_cfg.sampling_rate,
                        frame_length=feature_cfg.frame_length,
                        frame_shift=feature_cfg.frame_shift,
                        dither=feature_cfg.dither,
                        preemph_coeff=feature_cfg.preemph_coeff,
                        remove_dc_offset=feature_cfg.remove_dc_offset,
                        window_type=feature_cfg.window_type,
                        round_to_power_of_two=feature_cfg.round_to_power_of_two,
                        blackman_coeff=feature_cfg.blackman_coeff,
                        snip_edges=feature_cfg.snip_edges
                    ),
                    mel_opts=KaldifeatMelOptions(
                        num_bins=feature_cfg.num_mel_bins or 23,
                        low_freq=feature_cfg.low_freq,
                        high_freq=feature_cfg.high_freq,
                        vtln_low=feature_cfg.vtln_low,
                        vtln_high=feature_cfg.vtln_high,
                        debug_mel=feature_cfg.debug_mel,
                        htk_mode=feature_cfg.htk_mode,
                    ),
                    use_energy=feature_cfg.use_energy,
                    energy_floor=feature_cfg.energy_floor,  # default was 0.0
                    raw_energy=feature_cfg.raw_energy,
                    htk_compat=feature_cfg.htk_compat,
                    use_log_fbank=feature_cfg.use_log_fbank,
                    use_power=feature_cfg.use_power,
                    device=feature_cfg.device
                )
            )
        elif feature_cfg.feature_type == "mfcc":

            extractor = KaldifeatMfcc(
                KaldifeatMfccConfig(
                    frame_opts=KaldifeatFrameOptions(
                        sampling_rate=feature_cfg.sampling_rate,
                        frame_length=feature_cfg.frame_length,
                        frame_shift=feature_cfg.frame_shift,
                        dither=feature_cfg.dither,
                        preemph_coeff=feature_cfg.preemph_coeff,
                        remove_dc_offset=feature_cfg.remove_dc_offset,
                        window_type=feature_cfg.window_type,
                        round_to_power_of_two=feature_cfg.round_to_power_of_two,
                        blackman_coeff=feature_cfg.blackman_coeff,
                        snip_edges=feature_cfg.snip_edges
                    ),
                    mel_opts=KaldifeatMelOptions(
                        num_bins=feature_cfg.num_mel_bins or 23,
                        low_freq=feature_cfg.low_freq,
                        high_freq=feature_cfg.high_freq,
                        vtln_low=feature_cfg.vtln_low,
                        vtln_high=feature_cfg.vtln_high,
                        debug_mel=feature_cfg.debug_mel,
                        htk_mode=feature_cfg.htk_mode
                    ),

                    # sampling_rate=feature_cfg.sampling_rate,
                    num_ceps=feature_cfg.num_ceps,
                    use_energy=feature_cfg.use_energy,
                    energy_floor=feature_cfg.energy_floor,
                    raw_energy=feature_cfg.raw_energy,
                    cepstral_lifter=feature_cfg.cepstral_lifter,
                    htk_compat=feature_cfg.htk_compat,
                    device=feature_cfg.device,
                    chunk_size=feature_cfg.chunk_size
                )
            )
        else:
            raise ValueError(
                f"Unsupported feature type: {feature_cfg.feature_type}")

        # Compute and store features to per-dataset directory (multiprocessing-friendly)
        dataset_storage_path = Path(
            storage_root) / f"{dataset_name}"
        dataset_storage_path.mkdir(parents=True, exist_ok=True)

        storage_type_map: Dict[str, Union[type[LilcomChunkyWriter], type[LilcomFilesWriter], type[NumpyFilesWriter]]] = {
            "lilcom_chunky": LilcomChunkyWriter,
            "lilcom_files": LilcomFilesWriter,
            "numpy": NumpyFilesWriter,
        }
        storage_writer_cls = storage_type_map.get(
            feature_cfg.storage_type, LilcomChunkyWriter
        )

        # Manifest path for the cuts with features
        manifest_path = dataset_storage_path / \
            f"{dataset_name}_{split_name}_with_feats.jsonl.gz"

        # Compute and store features - writes to disk incrementally
        # Note: We don't store the return value to avoid loading everything into memory
        cuts.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=dataset_storage_path / split_name,
            storage_type=storage_writer_cls,
            num_workers=feature_cfg.num_workers or 4,
            manifest_path=manifest_path,
            batch_duration=feature_cfg.batch_duration,
            overwrite=feature_cfg.overwrite,
        )

        # Load the manifest lazily to avoid memory issues with large datasets
        # This only loads metadata, not the actual feature arrays
        print(f"  → Loading manifest lazily from {manifest_path}")
        cuts_with_feats = CutSet.from_jsonl_lazy(manifest_path)

        # Don't call describe() as it iterates through the entire dataset
        # and loads everything into memory. Users can call it manually if needed.
        print(f"  ✓ Features computed and cached successfully")

        return cuts_with_feats

    @staticmethod
    def _normalize_splits(
        dataset_cut_sets: Dict[str, CutSet],
        dataset_name: str,
    ) -> Dict[str, CutSet]:
        """
        Normalize dataset splits to unified format: train, val, test.

        Handles various split naming conventions:
        - dev → val
        - Maps test appropriately

        Note: Only uses externally defined splits. Does NOT auto-split to avoid
        materializing large CutSets in memory.

        Args:
            dataset_cut_sets: Dictionary of CutSets by split name
            dataset_name: Name of the dataset

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

        # Rename splits according to mapping
        for split_name, cuts in dataset_cut_sets.items():
            normalized_name = split_mapping.get(split_name.lower(), split_name)
            normalized[normalized_name] = cuts

        return normalized

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
            # Get parameters from data_loading config if available
            subsampling = getattr(
                data_loading, 'subsampling', 10) if data_loading else 10
            context_size = getattr(
                data_loading, 'context_size', 7) if data_loading else 7
            min_enroll_len = getattr(
                data_loading, 'min_enroll_len', 1.0) if data_loading else 1.0
            max_enroll_len = getattr(
                data_loading, 'max_enroll_len', 5.0) if data_loading else 5.0
            chunk_size = getattr(data_loading, 'chunk_size',
                                 None) if data_loading else None

            # chunk_size is now mandatory for ego-centric datasets
            if chunk_size is None:
                raise ValueError(
                    "chunk_size is required for ego-centric diarization datasets. Please specify chunk_size in data_loading config.")

            return EgoCentricDiarizationDataset(
                cuts=cuts,
                chunk_size=chunk_size,
                min_enroll_len=min_enroll_len,
                max_enroll_len=max_enroll_len,
                context_size=context_size,
                subsampling=subsampling,
            )
        elif label_type == "binary":
            return DiarizationDataset(cuts)
        else:
            raise ValueError(f"Unsupported label_type: {label_type}")

    @staticmethod
    def create_dataloader(
        cuts: CutSet,
        data_loading: DataLoadingConfig,
        batch_size: int,
        label_type: LabelType = "binary",
        random_seed: int = 42,
        shuffle: bool = True,
    ) -> DataLoader[Any]:
        """
        Create a PyTorch DataLoader from Lhotse CutSet for diarization.

        This method creates a standard PyTorch DataLoader compatible with Accelerate:
        - Chunks cuts into fixed-size segments for consistent batching
        - Uses standard PyTorch DataLoader with explicit batch_size
        - Fully compatible with Accelerate's prepare() method

        Args:
            cuts: Lhotse CutSet containing audio cuts with supervisions and precomputed features
            data_loading: Configuration for data loading (dataloader settings)
            batch_size: Explicit batch size for DataLoader (required for Accelerate compatibility)
            label_type: Type of labels ("ego" or "binary")
            random_seed: Random seed for reproducibility
            shuffle: Whether to shuffle the dataset

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
                batch_size=32,
                label_type="binary",
                random_seed=42,
            )
            ```
        """
        # Apply chunking if specified in config for consistent segment lengths
        if data_loading.chunk_size is not None and data_loading.chunk_size > 0:
            cuts = cuts.cut_into_windows(duration=data_loading.chunk_size)
            # Pad cuts to ensure all have the same duration
            cuts = cuts.pad(duration=data_loading.chunk_size)
            print(
                f"Applied chunking with window size {data_loading.chunk_size}s and padding")
        elif label_type == "ego":
            # For ego-centric dataset, chunking is mandatory
            raise ValueError(
                "chunk_size is required for ego-centric datasets. Please specify chunk_size in data_loading config.")

        # Create dataset
        dataset = DatasetManager._create_dataset(
            cuts=cuts, label_type=label_type, data_loading=data_loading)

        # Create worker init function for reproducibility
        worker_init_fn = make_worker_init_fn(seed=random_seed)

        # Set up collate function for ego-centric dataset
        collate_fn = None
        if label_type == "ego":
            collate_fn = EgoCentricDiarizationDataset.collate_fn

        # Create standard PyTorch DataLoader with explicit batch_size
        dataloader_cfg = data_loading.dataloader
        dataloader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            drop_last=True,  # Ensure consistent batch sizes
            persistent_workers=dataloader_cfg.persistent_workers if dataloader_cfg.num_workers > 0 else False,
            prefetch_factor=dataloader_cfg.prefetch_factor if dataloader_cfg.num_workers > 0 else None,
        )

        return dataloader

    @staticmethod
    def create_train_val_dataloaders(
        train_cuts: CutSet,
        val_cuts: Optional[CutSet],
        data_loading: DataLoadingConfig,
        batch_size: int,
        label_type: LabelType = "binary",
        random_seed: int = 42,
    ) -> Tuple[DataLoader[Any], Optional[DataLoader[Any]]]:
        """
        Create training and validation DataLoaders for diarization.

        Args:
            train_cuts: Training CutSet with precomputed features
            val_cuts: Validation CutSet with precomputed features (optional)
            data_loading: Configuration for data loading
            batch_size: Explicit batch size for DataLoaders (required for Accelerate compatibility)
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
                batch_size=32,
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
            batch_size=batch_size,
            label_type=label_type,
            random_seed=random_seed,
            shuffle=True,
        )

        val_dataloader = None
        if val_cuts:
            print("\n" + "=" * 60)
            print("Creating validation dataloader")
            print("=" * 60)

            val_dataloader = DatasetManager.create_dataloader(
                cuts=val_cuts,
                data_loading=data_loading,
                batch_size=batch_size,
                label_type=label_type,
                random_seed=random_seed,
                shuffle=False,  # No shuffling for validation
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

        # Get storage path from global_config if available
        global_config = getattr(dataset, "global_config", None)
        storage_path = Path(global_config.storage_path) if global_config and hasattr(
            global_config, 'storage_path') else None

        # Try to load existing manifests (prioritizing cached features)
        existing_manifests = DatasetManager._try_load_existing_manifests(
            output_dir, dataset.name, storage_path
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
                storage_path, split_name, dataset.name
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
        corpus_path = DatasetManager._download_dataset(
            dataset, download_function)

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

        # Step 4: Normalize split names (dev → val)
        dataset_cut_sets = DatasetManager._normalize_splits(
            dataset_cut_sets, dataset.name
        )

        # Step 5: Load or compute features
        dataset_cut_sets = DatasetManager._process_features_for_dataset(
            dataset, dataset_cut_sets
        )

        print(f"✓ {dataset.name} ready!\n")
        return dataset_cut_sets

    @staticmethod
    def _load_precomputed_dataset(
        dataset: Any,
        validation_split: float,
    ) -> Dict[str, CutSet]:
        """Load a dataset that already has prepared manifests on disk."""

        manifest_dir: Optional[Path] = None
        precomputed_dir = getattr(dataset, "precomputed_manifest_dir", None)
        if precomputed_dir is not None:
            manifest_dir = Path(precomputed_dir)
        else:
            process_kwargs = dataset.get_process_kwargs()
            output_dir = process_kwargs.get("output_dir") if isinstance(
                process_kwargs, dict) else None
            if output_dir is not None:
                manifest_dir = Path(output_dir)

        if manifest_dir is None:
            raise ValueError(
                f"Precomputed dataset {dataset.name} does not specify a manifest directory"
            )

        if not manifest_dir.exists():
            raise FileNotFoundError(
                f"Manifest directory {manifest_dir} for dataset {dataset.name} was not found"
            )

        print(f"\n{'=' * 60}")
        print(f"Loading precomputed dataset: {dataset.name}")
        print(f"{'=' * 60}")

        # Get storage path from global_config if available
        global_config = getattr(dataset, "global_config", None)
        storage_path = Path(global_config.storage_path) if global_config and hasattr(
            global_config, 'storage_path') else None

        manifests = DatasetManager._try_load_existing_manifests(
            manifest_dir, dataset.name, storage_path
        )

        if manifests is None:
            raise ValueError(
                f"Expected precomputed manifests for {dataset.name} in {manifest_dir}"
            )

        dataset_cut_sets = DatasetManager._manifests_to_cutsets_dict(
            manifests, dataset.name
        )

        dataset_cut_sets = DatasetManager._normalize_splits(
            dataset_cut_sets, dataset.name
        )

        dataset_cut_sets = DatasetManager._process_features_for_dataset(
            dataset, dataset_cut_sets
        )

        print(f"✓ {dataset.name} ready (loaded from cached manifests)!\n")
        return dataset_cut_sets

    @staticmethod
    def load_datasets(
        datasets: List[Any],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        validation_split: float = 0.1,
        test_split: float = 0
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
        all_cut_sets: Dict[str, Dict[str, CutSet]] = {}
        for dataset in params.datasets:
            if getattr(dataset, "precomputed_only", False):
                dataset_cut_sets = DatasetManager._load_precomputed_dataset(
                    dataset=dataset,
                    validation_split=params.validation_split,
                )
            else:
                process_function, download_function = import_recipe(
                    dataset.name
                )
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
                manifests_typed = cast(
                    Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]], manifests)
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
                nested_manifests = cast(
                    Dict[str, Dict[str, Any]], manifest_dict)
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
                recording_set = cast(
                    Optional[RecordingSet], recording_candidate)

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


# if __name__ == "__main__":
#     args, dataset_configs = datasets_manager_parser()

#     # Create LoadDatasetsParams with the parsed dataset configurations
#     load_params = LoadDatasetsParams(
#         datasets=dataset_configs,
#         batch_size=32,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#         validation_split=0.1,
#         test_split=0.1,
#     )

#     cut_sets = DatasetManager.load_datasets(**vars(load_params))

#     print(f"Loaded {len(cut_sets)} datasets")
#     for dataset_name, splits in cut_sets.items():
#         print(f"\nDataset: {dataset_name}")
#         for split_name, cut_set in splits.items():
#             print(f"  Split: {split_name}")
#             cut_set.describe()
