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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lhotse as lh
import torch
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    Mfcc,
    MfccConfig,
    RecordingSet,
    SupervisionSet,
    load_manifest
)
from lhotse.features.io import (
    LilcomChunkyWriter,
    LilcomFilesWriter,
    NumpyFilesWriter,
)
from data_manager import recipes
from data_manager.dataset_types import FeatureConfig, LoadDatasetsParams
from data_manager.parse_args import datasets_manager_parser


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
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Try to load existing manifests from disk to skip re-preparation.

        Args:
            output_dir: Base output directory for manifests
            dataset_name: Name of the dataset

        Returns:
            Dictionary of manifests by split if they exist, None otherwise
        """
        from lhotse import CutSet, RecordingSet, SupervisionSet

        # output_dir already includes dataset name (e.g., manifests/ava_avd)
        # So we use it directly, not append dataset_name again
        manifest_dir = output_dir
        if not manifest_dir.exists():
            return None

        # Look for standard manifest files
        # Common patterns:
        #   - recordings_train.jsonl.gz
        #   - ava_avd_recordings_train.jsonl.gz (dataset-prefixed)
        splits = set()
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
            def resolve_manifest(patterns, glob_suffix):
                for path in patterns:
                    if path.exists():
                        return path
                wildcard = next(
                    (p for p in manifest_dir.glob(f"*{glob_suffix}")),
                    None,
                )
                return wildcard

            recordings_path = resolve_manifest(
                recordings_patterns, f"recordings_{split_name}.jsonl.gz"
            )
            supervisions_path = resolve_manifest(
                supervisions_patterns, f"supervisions_{split_name}.jsonl.gz"
            )
            cuts_path = resolve_manifest(
                cuts_patterns, f"cuts_{split_name}.jsonl.gz"
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

        return manifests if manifests else None

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
            # Use load_manifest (eager) instead of load_manifest_lazy for BucketingSampler compatibility
            return load_manifest(cache_path)

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
            print(f"Resampling audio {cuts[0].sampling_rate} -> {eature_cfg.sampling_rate}")
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
                window_sec = float(feature_cfg.cut_window_seconds)
                # Use sliding windows with no overlap for simplicity; can be made configurable
                cuts = cuts.cut_into_windows(
                    duration=window_sec, hop=window_sec)
            except Exception as e:
                print(f"Warning: failed to window cuts: {e}")

        # Compute and store features to per-dataset directory (multiprocessing-friendly)
        dataset_storage_path = Path(storage_root) / dataset_name
        dataset_storage_path.mkdir(parents=True, exist_ok=True)

        storage_type_map = {
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

        cuts_with_feats = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=str(dataset_storage_path),
            storage_type=storage_writer_cls,
            num_jobs=num_jobs,
            mix_eagerly=feature_cfg.mix_eagerly,
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
        if cuts and any(cut.has_features for cut in cuts[: min(10, len(cuts))]):
            print(f"💾 Saving {split_name} cuts with features to {cache_path}")
            cuts.to_file(cache_path)
        else:
            print(
                f"⚠️  Warning: {split_name} cuts don't have features yet, skipping save"
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
        normalized = {}

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
            from lhotse import CutSet

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
    def load_datasets(**kwargs) -> Dict[str, Dict[str, CutSet]]:
        """
        Load datasets and convert all manifest formats to CutSets for diarization tasks.

        This method handles the complete pipeline from dataset configuration to CutSet generation:
        1. Downloads datasets if needed
        2. Processes datasets to generate manifests
        3. Converts manifests to Lhotse CutSets
        4. Returns structured dictionary of CutSets organized by dataset and split

        Args:
            **kwargs: Keyword arguments passed to LoadDatasetsParams, including:
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
        params = LoadDatasetsParams(**kwargs)
        recipes = [
            (import_recipe(dataset.name), dataset) for dataset in params.datasets
        ]
        all_cut_sets = {}

        for recipe in recipes:
            (process_function, download_function), dataset = recipe
            if not download_function and not process_function:
                raise ValueError(
                    f"Dataset {dataset.name} has no download or process function"
                )

            print(f"\n{'=' * 60}")
            print(f"Processing dataset: {dataset.name}")
            print(f"{'=' * 60}")

            # Download dataset if download function exists
            corpus_path = None
            if download_function:
                # Inspect download kwargs to determine target path and force flag
                dl_kwargs = dataset.get_download_kwargs()
                target_dir = dl_kwargs.get("target_dir")
                force_dl = dl_kwargs.get("force_download", False)

                # If the target directory already exists and user did not request a
                # forced re-download, skip calling the download function.
                if target_dir:
                    try:
                        target_path = Path(target_dir)
                        if target_path.exists() and not force_dl:
                            print(
                                f"→ Skipping download for {dataset.name}: target_dir {target_path} exists (force_download={force_dl})"
                            )
                            corpus_path = target_path
                        else:
                            corpus_path = download_function(**dl_kwargs)
                    except Exception:
                        # If anything goes wrong while probing the path, fall back to
                        # calling the download function so the dataset can still be
                        # obtained (safer default than silent failure).
                        corpus_path = download_function(**dl_kwargs)
                else:
                    # No explicit target_dir provided by DatasetConfig; call download
                    corpus_path = download_function(**dl_kwargs)

            # Process dataset to get manifests
            if process_function:
                # Use the actual corpus path returned by download function if available
                process_kwargs = dataset.get_process_kwargs()
                if corpus_path and "corpus_dir" not in process_kwargs:
                    process_kwargs["corpus_dir"] = corpus_path
                elif corpus_path and "corpus_dir" in process_kwargs:
                    # Override with actual path if download function provided one
                    process_kwargs["corpus_dir"] = corpus_path

                sig = inspect.signature(process_function)
                valid_keys = set(sig.parameters.keys())

                if corpus_path:
                    if "audio_dir" in valid_keys:
                        process_kwargs["audio_dir"] = corpus_path
                    elif "data_dir" in valid_keys:
                        process_kwargs["data_dir"] = corpus_path
                    elif "corpus_dir" in valid_keys:
                        process_kwargs["corpus_dir"] = corpus_path

                filtered_kwargs = {
                    k: v for k, v in process_kwargs.items() if k in valid_keys
                }

                if len(filtered_kwargs) < len(process_kwargs):
                    ignored_keys = set(process_kwargs.keys()) - valid_keys
                    # If the user provided a generic 'corpus_dir', remap it where possible
                    if "corpus_dir" in ignored_keys:
                        if "audio_dir" in valid_keys:
                            filtered_kwargs["audio_dir"] = process_kwargs["corpus_dir"]
                            print(
                                "Note: 'corpus_dir' renamed to 'audio_dir' for this recipe.")
                            ignored_keys.remove("corpus_dir")
                        elif "data_dir" in valid_keys:
                            filtered_kwargs["data_dir"] = process_kwargs["corpus_dir"]
                            print(
                                "Note: 'corpus_dir' renamed to 'data_dir' for this recipe.")
                            ignored_keys.remove("corpus_dir")

                    if ignored_keys:
                        print(
                            f"⚠️  Warning: Ignoring unsupported process parameters for {dataset.name}: {ignored_keys}"
                        )

                output_dir = Path(process_kwargs.get(
                    "output_dir", "./manifests"))

                # Check if we can load existing manifests to skip preparation
                existing_manifests = DatasetManager._try_load_existing_manifests(
                    output_dir, dataset.name
                )

                if existing_manifests:
                    print(
                        f"✓ Using existing manifests for {dataset.name} (skip audio extraction & manifest creation)"
                    )
                    manifests = existing_manifests
                else:
                    print(
                        f"→ Preparing {dataset.name} dataset (manifests not found, will extract audio & create manifests)"
                    )
                    manifests = process_function(**filtered_kwargs)

                # Convert manifests to structured CutSet dictionary
                dataset_cut_sets = DatasetManager._manifests_to_cutsets_dict(
                    manifests, dataset.name
                )

                # Normalize split names (dev → val, auto-split if needed)
                val_split_ratio = (
                    params.validation_split
                    if hasattr(params, "validation_split")
                    else 0.1
                )
                dataset_cut_sets = DatasetManager._normalize_splits(
                    dataset_cut_sets, dataset.name, val_split_ratio
                )

                # Try to load or compute features depending on data loading strategy
                data_loading = getattr(
                    dataset.global_config, "data_loading", None)
                dl_strategy = (
                    data_loading.strategy
                    if data_loading is not None
                    else "precomputed_features"
                )

                if dl_strategy == "precomputed_features":
                    print(f"\nChecking feature cache for {dataset.name}...")
                    # Get feature storage path from global config and append dataset name
                    base_storage_path = dataset.global_config.storage_path
                    if base_storage_path:
                        # Append dataset name to create dataset-specific cache directory
                        for split_name, cuts in dataset_cut_sets.items():
                            storage_path = Path(
                                base_storage_path) / dataset.name
                            cached_cuts = (
                                DatasetManager._load_cached_cuts_with_features(
                                    storage_path, split_name
                                )
                            )
                            if cached_cuts is not None:
                                dataset_cut_sets[split_name] = cached_cuts
                            else:
                                # Compute features now and cache them for this split
                                print(
                                    f"  → {split_name}: Computing features and caching to {storage_path}"
                                )
                                dataset_cut_sets[split_name] = (
                                    DatasetManager._compute_and_cache_features_for_split(
                                        cuts,
                                        dataset.name,
                                        split_name,
                                        dataset.global_config.get_feature_config(),
                                        Path(base_storage_path),
                                    )
                                )
                    else:
                        print(
                            "  → No storage_path configured, features will be extracted on demand"
                        )
                else:
                    # On-the-fly/audio_samples strategies: skip precomputation entirely
                    print(
                        f"\nData loading strategy '{dl_strategy}' selected — skipping feature precomputation."
                    )

                all_cut_sets[dataset.name] = dataset_cut_sets
                print(f"✓ {dataset.name} ready!\n")

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
            if all(isinstance(v, dict) for v in manifests.values()):
                # Nested dict format - has splits with manifests
                result = {}
                for split_name, split_manifests in manifests.items():
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
                for v in manifests.values()
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
        cut_sets = []

        if manifests is None:
            raise ValueError(f"Dataset {dataset_name} has no manifests")

        # Handle Tuple format: (RecordingSet, SupervisionSet)
        if isinstance(manifests, tuple) and len(manifests) == 2:
            recording_set, supervision_set = manifests
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
            # Check if it's a nested dict (split-based): Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]
            if all(isinstance(v, dict) for v in manifests.values()):
                # Nested dict format - iterate through splits
                for split_name, split_manifests in manifests.items():
                    split_cut_sets = DatasetManager._manifests_to_cutsets(
                        split_manifests, f"{dataset_name}_{split_name}"
                    )
                    cut_sets.extend(split_cut_sets)

            # Check if it's a flat dict: Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]
            elif any(
                isinstance(v, (RecordingSet, SupervisionSet, CutSet))
                for v in manifests.values()
            ):
                recording_set = manifests.get(
                    "recordings", manifests.get("recording", None)
                )
                supervision_set = manifests.get(
                    "supervisions", manifests.get("supervision", None)
                )

                # Check if we have a CutSet directly
                if isinstance(manifests.get("cuts"), CutSet):
                    cut_sets.append(manifests["cuts"])
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
            if hasattr(manifests, "recordings") and hasattr(manifests, "supervisions"):
                cut_sets.append(
                    CutSet.from_manifests(
                        recordings=manifests.recordings,
                        supervisions=manifests.supervisions,
                    )
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} returned unrecognized manifest format: {type(manifests)}"
                )

        if not cut_sets:
            raise ValueError(
                f"Dataset {dataset_name} could not be converted to CutSet(s)"
            )

        return cut_sets


if __name__ == "__main__":
    """
        Example usage
        python datasets/data_manager.py --config configs/test_data.yml
    """
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

    print(f"Loaded {len(cut_sets)} cut sets")
    for cut_set in cut_sets:
        cut_set.describe()
