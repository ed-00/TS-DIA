#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Dataset Management and Processing (Distributed-Safe)

This module provides the core dataset management functionality for the TS-DIA project.
It handles dataset downloading, processing, and conversion to Lhotse CutSets for
diarization tasks, with support for 50+ speech datasets via Lhotse integration.

This class is designed to be instantiated with an Accelerate instance to
ensure distributed-safe data preparation, following the
"Main Process Prepares, All Processes Load" pattern.

Main Classes:
    DatasetManager: Main class for loading and processing datasets

Usage Examples:
    ```python
    from datasets import DatasetManager, parse_dataset_configs
    from accelerate import Accelerator

    accelerator = Accelerator()

    # Instantiate the manager with the accelerator
    data_manager = DatasetManager(accelerator=accelerator)

    # Load datasets from configuration
    configs = parse_dataset_configs('configs/my_datasets.yml')
    cut_sets = data_manager.load_datasets(datasets=configs)

    # Create dataloaders
    train_loader = data_manager.create_training_dataloader(...)
    val_loaders = data_manager.create_validation_dataloaders(...)
    ```
"""
import inspect
from pathlib import Path

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import json
import numpy as np


import multiprocessing as mp

import lhotse as lh
from lhotse import (
    CutSet,
    KaldifeatFbankConfig,
    KaldifeatFbank,
    KaldifeatMfcc,
    KaldifeatMfccConfig,
    RecordingSet,
    SupervisionSet,

)
from lhotse.cut import MonoCut

from lhotse.utils import (
    Pathlike
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
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator

from data_manager import recipes
from data_manager.dataset_types import (
    FeatureConfig,
    LoadDatasetsParams,
    DataLoadingConfig,
    LabelType,
)
from training.ego_dataset import EgoCentricDiarizationDataset
from utility.dataset_utils import prepare_training_cuts
from training import TrainingConfig, TrainingDatasetMap
from data_manager import GlobalConfig
from lhotse import load_manifest, store_manifest
import torch

# These functions are stateless and can be defined at the module level.


def __is_custom_recipe(dataset_name: str) -> bool:
    """Check if a custom recipe exists for the dataset."""
    download_function_name = f"download_{dataset_name}"
    process_function_name = f"prepare_{dataset_name}"
    return (
        download_function_name in recipes.__all__
        and process_function_name in recipes.__all__
    )


def __is_implemented_dataset(dataset_name: str) -> bool:
    """Check if a dataset is implemented in Lhotse."""
    download_function_name = f"download_{dataset_name}"
    process_function_name = f"prepare_{dataset_name}"
    return (
        download_function_name in lh.recipes.__all__
        and process_function_name in lh.recipes.__all__
    )


def __is_divertion_from_standard(dataset_name: str) -> bool:
    """Checks for diffrent yet uncommon naming conventions that exist"""
    if dataset_name == "voxceleb1" or dataset_name == "voxceleb2":
        return True
    return False


def fetch_diversion(
    dataset_name: str
) -> Tuple[
    Callable[..., Any],
    Optional[Callable[..., Any]],
]:
    """Fetches dataset process function that diviate from the normal naming"""
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
    Callable[..., Any],
    Optional[Callable[..., Any]],
]:
    """A Function that selects the correct recipe for the dataset"""
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
    """List all available datasets from both Lhotse and custom recipes."""
    clean_names_lhotse = set(
        name.replace("download_", "").replace("prepare_", "")
        for name in lh.recipes.__all__
    )
    custom_recipes = set(
        name.replace("download_", "").replace("prepare_", "")
        for name in recipes.__all__
    )
    all_datasets = clean_names_lhotse.union(custom_recipes)
    if not custom_recipes:
        print(
            "Note: Custom recipes package is available but empty (ready for expansion)"
        )
    return all_datasets


def import_recipe(
    dataset_name: str,
) -> Tuple[
    Callable[..., Any],
    Optional[Callable[..., Any]],
]:
    """A function to import the correct recipe for each dataset"""
    if dataset_name in list_available_datasets():
        return select_recipe(dataset_name)
    else:
        raise ValueError(
            f"Dataset {dataset_name} is not implemented, double check the dataset name and the recipe, available datasets: {list_available_datasets()}"
        )


def resolve_manifest(patterns: List[Path], glob_suffix: str, manifest_dir: Path) -> Optional[Path]:
    """Resolve manifest file path from patterns or glob search."""
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

    This class provides the main interface for loading and processing datasets
    in a distributed-safe manner. It must be initialized with an `Accelerator`
    instance.
    """

    def _try_load_existing_manifests(
        self, output_dir: Path, dataset_name: str, storage_path: Optional[Path] = None
    ) -> Optional[Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]]:
        """
        Try to load existing manifests from disk to skip re-preparation.
        Prioritizes cached CutSets with features if available.

        This method is only called by the main process.
        """

        # First, try to load cached CutSets with features if storage_path is provided
        if storage_path is not None:
            print("="*60)
            print(
                f"Checking for cached CutSets with features in {storage_path}")
            print("="*60)

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

                    if dataset_name not in manifest_file.parts and not manifest_file.name.startswith(f"{dataset_name}_"):
                        continue

                    filename = manifest_file.name.replace(".jsonl.gz", "")

                    start_pos = len(dataset_name) + 1
                    end_pos = filename.rfind("_with_feats")
                    if end_pos == -1:
                        continue
                    split_name = filename[start_pos:end_pos]

                    if not split_name:
                        continue

                    try:
                        cuts = CutSet.from_jsonl_lazy(manifest_file)
                    except Exception as exc:
                        print(
                            f"→ Failed to load cached CutSet from {manifest_file}: {exc}"
                        )
                        continue

                    seen_files.add(manifest_file)
                    cached_entry = cached_manifests.setdefault(split_name, {})
                    cached_entry["cuts"] = cuts
                    print(
                        f"✓ {split_name}: Loaded cached CutSet with features (lazy) from {manifest_file}"
                    )

            if cached_manifests:
                print(
                    f"✓ Using cached CutSets with features for {dataset_name} (skip manifest loading)")
                print("="*60)
                return cast(
                    Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]],
                    cached_manifests,
                )

        # Fall back to loading raw manifests
        manifest_dir = output_dir
        if not manifest_dir.exists():
            print("="*60)
            print(
                f"Manifest dir {manifest_dir} does not exist!")
            print("="*60)
            return None

        splits: set[str] = set()
        recordings_files = list(manifest_dir.glob("*recordings_*.jsonl.gz"))

        for file in recordings_files:
            filename = file.stem.replace(".jsonl", "")
            if "_recordings_" in filename:
                split_name = filename.split("_recordings_")[1]
                splits.add(split_name)
            elif filename.startswith("recordings_"):
                split_name = filename[len("recordings_"):]
                splits.add(split_name)

        if not splits:
            return None

        manifests = {}
        for split_name in splits:
            split_manifests = {}

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
            print(
                f"Resolved Manifests for split: {split_name}")
            if not (recordings_path or cuts_path):
                continue

            if recordings_path:
                split_manifests["recordings"] = RecordingSet.from_jsonl_lazy(
                    recordings_path)
                print(
                    f"- Loaded recordings (lazy) from: {recordings_path}")

            if supervisions_path:
                split_manifests["supervisions"] = SupervisionSet.from_jsonl_lazy(
                    supervisions_path
                )
                print(
                    f"- Loaded supervision (lazy) from: {supervisions_path}")

            if cuts_path:
                split_manifests["cuts"] = CutSet.from_jsonl_lazy(cuts_path)
                print(f"- Loaded cutset (lazy): {cuts_path}")

            if split_manifests:
                manifests[split_name] = split_manifests
            print("="*60)

        if not manifests:
            return None

        return cast(
            Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]],
            manifests,
        )

    def _load_cached_cuts_with_features(
        self, storage_path: Path, split_name: str, dataset_name: str
    ) -> Optional[CutSet]:
        """
        Load cached CutSet with pre-computed features. (Main process only)
        """
        cache_path = storage_path / \
            f"{dataset_name}_{split_name}_with_feats.jsonl.gz"

        if cache_path.exists():
            print(
                f"✓ {split_name}: Loaded from cache (lazy loading)")
            return CutSet.from_jsonl_lazy(cache_path)

        print(
            f"→ {split_name}: No cache found (will extract features)")
        return None

    def _compute_and_cache_features_for_split(
        self,
        cuts: CutSet,
        dataset_name: str,
        split_name: str,
        feature_cfg: FeatureConfig,
        storage_root: Path,

    ) -> CutSet:
        """
        Compute features for a single split and cache. (Main process only)
        """
        first_cut = next(iter(cuts))
        if first_cut.sampling_rate != feature_cfg.sampling_rate:
            print(
                f"Resampling audio {first_cut.sampling_rate} -> {feature_cfg.sampling_rate}")
            cuts = cuts.resample(feature_cfg.sampling_rate)

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
                    energy_floor=feature_cfg.energy_floor,
                    raw_energy=feature_cfg.raw_energy,
                    htk_compat=feature_cfg.htk_compat,
                    use_log_fbank=feature_cfg.use_log_fbank,
                    use_power=feature_cfg.use_power,
                    device=feature_cfg.device  # This will be 'cpu' or 'cuda'
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

        dataset_storage_path = Path(storage_root) / f"{dataset_name}"
        dataset_storage_path.mkdir(parents=True, exist_ok=True)

        storage_type_map: Dict[str, Union[type[LilcomChunkyWriter], type[LilcomFilesWriter], type[NumpyFilesWriter]]] = {
            "lilcom_chunky": LilcomChunkyWriter,
            "lilcom_files": LilcomFilesWriter,
            "numpy": NumpyFilesWriter,
        }
        storage_writer_cls = storage_type_map.get(
            feature_cfg.storage_type, LilcomChunkyWriter
        )

        manifest_path = dataset_storage_path / \
            f"{dataset_name}_{split_name}_with_feats.jsonl.gz"

        cuts.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=dataset_storage_path / split_name,
            storage_type=storage_writer_cls,
            num_workers=feature_cfg.num_workers or 4,
            manifest_path=manifest_path,
            batch_duration=feature_cfg.batch_duration,
            overwrite=feature_cfg.overwrite,
        )

        print(
            f"→ Loading manifest lazily from {manifest_path}")
        cuts_with_feats = CutSet.from_jsonl_lazy(manifest_path)

        print(f"✓ Features computed and cached successfully")
        return cuts_with_feats

    def _normalize_splits(
        self,
        dataset_cut_sets: Dict[str, CutSet],
        dataset_name: str,
    ) -> Dict[str, CutSet]:
        """
        Normalize dataset splits to unified format: train, val, test. (Main process only)
        """
        normalized: Dict[str, Any] = {}
        split_mapping = {
            "dev": "val",
            "development": "val",
            "validation": "val",
            "train": "train",
            "test": "test",
            "eval": "test",
        }

        for split_name, cuts in dataset_cut_sets.items():
            normalized_name = split_mapping.get(split_name.lower(), split_name)
            normalized[normalized_name] = cuts

        return normalized

    def _create_dataset(
        self,
        cuts: CutSet,
        label_type: LabelType = "binary",
        data_loading: Optional[DataLoadingConfig] = None,

    ) -> Union[DiarizationDataset, Any]:
        """
        Create a diarization dataset based on the label type.
        """
        if label_type == "ego":

            subsampling = getattr(
                data_loading, 'subsampling', 10) if data_loading else 10
            context_size = getattr(
                data_loading, 'context_size', 7) if data_loading else 7
         
            return EgoCentricDiarizationDataset(
                cuts=cuts,
                context_size=context_size,
                subsampling=subsampling,
        
            )
        elif label_type == "binary":
            return DiarizationDataset(cuts)
        else:
            raise ValueError(f"Unsupported label_type: {label_type}")

    def create_validation_dataloaders(
        self,
        cut_sets: Dict[str, Dict[str, CutSet]],
        dataset_configs: List[Any],
        global_config: GlobalConfig,
        training_config: TrainingConfig,
        accelerator: Accelerator,
        label_type: LabelType = "binary",
        random_seed: int = 42,
    ) -> Dict[str, Tuple[DataLoader[Any], int]]:
        """
        Create validation dataloaders based on training configuration.
        """
        val_dataloaders: Dict[str, Tuple[DataLoader[Any], int]] = {}

        if not training_config.validation or not training_config.validation.validation_dataset_map:
            return val_dataloaders

        # Get cache directory from global config
        if not global_config.cache_dir:
            raise ValueError(
                "global_config.cache_dir must be set for caching.")
        final_cache_dir = Path(global_config.cache_dir)

        validation_map = training_config.validation.validation_dataset_map

        if validation_map.combine:
            print(
                "→ Combining pre-processed validation splits...")
            val_cuts = prepare_training_cuts(cut_sets, validation_map)
            print(
                f"✓ Combined {len(val_cuts)} validation cuts")
            # Aggregate metadata sizes across validation splits, fall back to compute
            dataset_size = 0
            for split_info in validation_map.splits:
                size = self._get_split_dataset_size_from_cache_or_compute(
                    cut_sets=cut_sets,
                    cache_dir=final_cache_dir,
                    dataset_name=split_info.dataset_name,
                    split_name=split_info.split_name,
                    label_type=label_type,
                    accelerator=accelerator
                )
                dataset_size += size

            val_dataloaders["val"] = (self.create_dataloader(
                cuts=val_cuts,
                data_loading=global_config.data_loading,
                batch_size=training_config.batch_size,
                label_type=label_type,
                random_seed=random_seed,
                cache_dir=final_cache_dir,
            ), dataset_size)
        else:
            print(
                "→ Preparing separate validation dataloaders...")
            for split_info in validation_map.splits:
                val_key = f"{split_info.dataset_name}_{split_info.split_name}"
                print(f"  - {val_key}")

                # Create a temporary map for this specific split
                temp_map = TrainingDatasetMap(
                    combine=False, splits=[split_info])

                val_cuts = prepare_training_cuts(
                    cut_sets=cut_sets,
                    training_dataset_map=temp_map
                )
                dataset_size = self._get_split_dataset_size_from_cache_or_compute(
                    cut_sets=cut_sets,
                    cache_dir=final_cache_dir,
                    dataset_name=split_info.dataset_name,
                    split_name=split_info.split_name,
                    label_type=label_type,
                    accelerator=accelerator
                )

                val_dataloaders[val_key] = (self.create_dataloader(
                    cuts=val_cuts,
                    data_loading=global_config.data_loading,
                    batch_size=training_config.batch_size,
                    label_type=label_type,
                    random_seed=random_seed,
                    cache_dir=final_cache_dir,
                ), dataset_size)

        return val_dataloaders

    # def __save_index_map(
    #     self,
    #     index_map:  List[Tuple[int, str | None]],
    #     index_path: Path,
    # ) -> None:
    #     """
    #     Save the cut to speaker index map to disk.
    #     """
    #     print(f"Saving index map to {index_path}...")
    #     np.save(index_path, [(idx, spk if spk is not None else "")
    #             for idx, spk in index_map], allow_pickle=True)

    # def __load_index_map(
    #     self,
    #     index_path: Path,
    # ) -> List[Tuple[int, str | None]]:
    #     """
    #     Load the cut to speaker index map from disk.
    #     """
    #     raw_data = np.load(index_path, allow_pickle=True)
    #     return [(int(idx), spk if spk != "" else None) for idx, spk in raw_data]

    def create_training_dataloader(
        self,
        cut_sets: Dict[str, Dict[str, CutSet]],
        global_config: GlobalConfig,
        training_config: TrainingConfig,
        accelerator: Accelerator,
        random_seed: int = 42,
    ) -> Tuple[DataLoader[Any], int]:
        """
        Create training dataloader by combining pre-processed splits.
        Note: Subsetting, windowing, and labels are already applied during load_datasets.
        """
        assert training_config.training_dataset_map is not None, "Training config map must be provided"

        # Get cache directory from global config
        if not global_config.cache_dir:
            raise ValueError(
                "global_config.cache_dir must be set for caching.")
        final_cache_dir = Path(global_config.cache_dir)

        # Combine the pre-processed training splits directly (no intermediate cache)
        print("→ Combining pre-processed training splits...")
        train_cuts = prepare_training_cuts(
            cut_sets, training_config.training_dataset_map)

        if train_cuts is None:
            raise ValueError("No training data found for dataset.")

        print(f"✓ Combined {len(train_cuts)} training cuts")

        dataset = EgoCentricDiarizationDataset(
            cuts=train_cuts,
            context_size=training_config.eval_knobs.get("context_size", 7),
            subsampling=training_config.eval_knobs.get("subsampling", 10),
        )

        # Attempt to compute total dataset size using cached metadata per split
        total_data_size = 0
        training_map = training_config.training_dataset_map
        if training_map and hasattr(training_map, "splits"):
            for split_info in training_map.splits:
                size = self._get_split_dataset_size_from_cache_or_compute(
                    cut_sets=cut_sets,
                    cache_dir=final_cache_dir,
                    dataset_name=split_info.dataset_name,
                    split_name=split_info.split_name,
                    label_type="ego",
                    accelerator=accelerator
                )
                total_data_size += size
        else:
            # Fallback: compute from combined cuts if mapping is not available
            total_data_size = EgoCentricDiarizationDataset.get_total_dataset_size(
                train_cuts, desc=f"Calculating total training dataset size {accelerator.process_index}")
        accelerator.wait_for_everyone()

        return (DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            num_workers=global_config.data_loading.dataloader.num_workers,
            worker_init_fn=make_worker_init_fn(seed=random_seed),
            collate_fn=EgoCentricDiarizationDataset.collate_fn,
        ), total_data_size)

    def create_dataloader(
        self,
        cuts: CutSet,
        data_loading: DataLoadingConfig,
        batch_size: int,
        label_type: LabelType = "binary",
        random_seed: int = 42,
        cache_dir: Optional[Path] = None,
    ) -> DataLoader[Any]:
        """
        Create a PyTorch DataLoader from Lhotse CutSet for diarization.

        The accelerator instance is passed during dataset creation.
        """

        dataset = self._create_dataset(
            cuts=cuts, label_type=label_type, data_loading=data_loading)

        worker_init_fn = make_worker_init_fn(seed=random_seed)

        collate_fn = None
        if label_type == "ego":
            collate_fn = EgoCentricDiarizationDataset.collate_fn

        dataloader_cfg = data_loading.dataloader

        # IterableDataset-specific handling
        is_iterable = isinstance(dataset, IterableDataset)

        if is_iterable and getattr(dataloader_cfg, "shuffle", False):
            print(
                "⚠️  Warning: 'shuffle' is ignored for IterableDataset — the dataset implements its own shuffle buffer."
            )

        # For iterable datasets, avoid dropping the last (partially-filled) batch by default
        drop_last = False if is_iterable else True

        dataloader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=dataloader_cfg.num_workers,
            pin_memory=dataloader_cfg.pin_memory,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=dataloader_cfg.persistent_workers if dataloader_cfg.num_workers > 0 else False,
            prefetch_factor=dataloader_cfg.prefetch_factor if dataloader_cfg.num_workers > 0 else None,
        )

        return dataloader

    def _download_dataset(
        self,
        dataset: Any,
        download_function: Optional[Callable[..., Union[Path, None, Any]]],
    ) -> Optional[Path]:
        """
        Download a dataset if needed. (Main process only)
        """
        if not download_function:
            return None

        dl_kwargs = dataset.get_download_kwargs()
        target_dir = dl_kwargs.get("target_dir")
        force_dl = dl_kwargs.get("force_download", False)

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
                return download_function(**dl_kwargs)
        else:
            return download_function(**dl_kwargs)

    def _prepare_process_kwargs(
        self,
        dataset: Any,
        corpus_path: Optional[Path],
        process_function: Callable[..., Any],
    ) -> Dict[str, Any]:
        """
        Prepare and validate kwargs for the process function. (Main process only)
        """
        process_kwargs = dataset.get_process_kwargs()

        if corpus_path and "corpus_dir" not in process_kwargs:
            process_kwargs["corpus_dir"] = corpus_path
        elif corpus_path and "corpus_dir" in process_kwargs:
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
                    f"⚠️  Warning: Ignoring unsupported process parameters "
                    f"for {dataset.name}: {ignored_keys}"
                )

        return filtered_kwargs

    def _load_or_prepare_manifests(
        self,
        dataset: Any,
        process_function: Callable[..., Any],
        process_kwargs: Dict[str, Any],
    ) -> Any:
        """
        Load existing manifests or prepare new ones. (Main process only)
        """
        output_dir = Path(process_kwargs.get("output_dir", "./manifests"))

        global_config = getattr(dataset, "global_config", None)
        storage_path = Path(global_config.storage_path) if global_config and hasattr(
            global_config, 'storage_path') else None

        existing_manifests = self._try_load_existing_manifests(
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

    def _process_features_for_dataset(
        self,
        dataset: Any,
        dataset_cut_sets: Dict[str, CutSet],
    ) -> Dict[str, CutSet]:
        """
        Load cached features or compute new ones. (Main process only)
        """
        global_config = getattr(dataset, "global_config", None)
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

    def _manifests_to_cutsets_dict(
        self, manifests: Any, dataset_name: str
    ) -> Dict[str, CutSet]:
        """
        Convert any manifest format to a dictionary of CutSets. (Main process only)
        """
        if manifests is None:
            raise ValueError(f"Dataset {dataset_name} has no manifests")

        if isinstance(manifests, dict):
            manifest_dict = cast(Dict[str, Any], manifests)
            if all(isinstance(v, dict) for v in manifest_dict.values()):
                result: Dict[str, CutSet] = {}
                manifests_typed = cast(
                    Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]], manifests)
                for split_name, split_manifests in manifests_typed.items():
                    cut_sets_list = self._manifests_to_cutsets(
                        split_manifests, f"{dataset_name}_{split_name}"
                    )
                    if cut_sets_list:
                        result[split_name] = cut_sets_list[0]
                return result

            elif any(
                isinstance(v, (RecordingSet, SupervisionSet, CutSet))
                for v in cast(Dict[str, Any], manifests).values()
            ):
                cut_sets_list = self._manifests_to_cutsets(
                    manifests, dataset_name
                )
                return {"train": cut_sets_list[0]} if cut_sets_list else {}

        cut_sets_list = self._manifests_to_cutsets(
            manifests, dataset_name)
        if not cut_sets_list:
            return {}

        if len(cut_sets_list) == 1:
            return {"train": cut_sets_list[0]}
        else:
            split_names = ["train", "dev", "test"]
            return {
                split_names[i] if i < len(split_names) else f"split_{i}": cutset
                for i, cutset in enumerate(cut_sets_list)
            }

    def _manifests_to_cutsets(self, manifests: Any, dataset_name: str) -> List[CutSet]:
        """Convert any manifest format to CutSet(s). (Main process only)"""
        cut_sets: List[CutSet] = []

        if manifests is None:
            raise ValueError(f"Dataset {dataset_name} has no manifests")

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

        elif isinstance(manifests, (RecordingSet, SupervisionSet)):
            if isinstance(manifests, RecordingSet):
                cut_sets.append(
                    CutSet.from_manifests(
                        recordings=manifests,
                        supervisions=None,
                    )
                )
            else:
                raise ValueError(
                    f"Dataset {dataset_name} returned only SupervisionSet without recordings"
                )

        elif isinstance(manifests, dict):
            manifest_dict = cast(Dict[str, Any], manifests)
            if all(isinstance(v, dict) for v in manifest_dict.values()):
                nested_manifests = cast(
                    Dict[str, Dict[str, Any]], manifest_dict)
                for split_name, split_manifests in nested_manifests.items():
                    split_cut_sets = self._manifests_to_cutsets(
                        split_manifests, f"{dataset_name}_{split_name}"
                    )
                    cut_sets.extend(split_cut_sets)

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

        else:
            if not isinstance(manifests, tuple) and hasattr(manifests, "recordings") and hasattr(manifests, "supervisions"):
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

    def _process_single_dataset(
        self,
        dataset: Any,
        process_function: Optional[Callable[..., Any]],
        download_function: Optional[Callable[..., Union[Path, None, Any]]],
    ) -> Dict[str, CutSet]:
        """
        Process a single dataset: download, manifests, features. (Main process only)
        """
        if download_function is None and process_function is None:
            raise ValueError(
                f"Dataset {dataset.name} has no download or process function"
            )

        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset.name}")
        print(f"{'=' * 60}")

        corpus_path = self._download_dataset(
            dataset, download_function)

        if process_function is None:
            raise ValueError(f"Dataset {dataset.name} has no process function")

        process_kwargs = self._prepare_process_kwargs(
            dataset, corpus_path, process_function
        )
        manifests = self._load_or_prepare_manifests(
            dataset, process_function, process_kwargs
        )

        dataset_cut_sets = self._manifests_to_cutsets_dict(
            manifests, dataset.name
        )

        dataset_cut_sets = self._normalize_splits(
            dataset_cut_sets, dataset.name
        )

        dataset_cut_sets = self._process_features_for_dataset(
            dataset, dataset_cut_sets
        )

        print(f"✓ {dataset.name} ready (raw manifests)!\n")
        return dataset_cut_sets

    def _check_final_cache(
        self,
        cache_dir: Path,
        dataset_name: str,
        split_name: str,
    ) -> Optional[Path]:
        """
        Check if fully processed cache exists. (Safe for all processes)
        """
        cache_path = cache_dir / dataset_name / split_name / "cuts_windowed.jsonl.gz"
        return cache_path if cache_path.exists() else None

    def _save_to_final_cache(
        self,
        cuts: CutSet,
        cache_dir: Path,
        dataset_name: str,
        split_name: str,
    ) -> None:
        """
        Save fully processed cuts to final cache. (Main process only)
        """
        cache_path = cache_dir / dataset_name / split_name
        cache_path.mkdir(parents=True, exist_ok=True)

        output_file = cache_path / "cuts_windowed.jsonl.gz"

        # Ensure cuts are materialized (not lazy) before saving
        if hasattr(cuts, 'is_lazy') and cuts.is_lazy:
            print(
                f"→ Materializing lazy CutSet before saving...")
            cuts = cuts.to_eager()

        print(
            f"→ Saving {len(cuts)} cuts to {output_file}...")
        cuts.to_file(output_file)

        # Verify the file was written successfully
        if output_file.exists() and output_file.stat().st_size > 0:
            print(
                f"✓ Saved final cache: {output_file} ({output_file.stat().st_size / (1024*1024):.2f} MB)")
        else:
            raise RuntimeError(
                f"Failed to save cache file or file is empty: {output_file}")

    def _save_cache_metadata(
        self,
        cache_dir: Path,
        dataset_name: str,
        split_name: str,
        dataset_size: Optional[int] = None,
        num_cuts: Optional[int] = None,
        label_type: Optional[LabelType] = None,
    ) -> None:
        """
        Save small JSON metadata alongside final cache to avoid re-computing
        dataset sizes repeatedly (main process only).
        """
        metadata_dir = cache_dir / dataset_name / split_name
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / "cache_metadata.json"
        meta = {
            "dataset_name": dataset_name,
            "split_name": split_name,
            "dataset_size": int(dataset_size) if dataset_size is not None else None,
            "num_cuts": int(num_cuts) if num_cuts is not None else None,
            "label_type": label_type if label_type is not None else None,
            "format_version": 1,
        }
        try:
            with open(metadata_file, "w") as fh:
                json.dump(meta, fh, indent=2)
            print(f"✓ Saved cache metadata: {metadata_file}")
        except Exception as exc:
            print(
                f"⚠️  Warning: Failed to write cache metadata to {metadata_file}: {exc}")

    def _load_cache_metadata(
        self,
        cache_dir: Path,
        dataset_name: str,
        split_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load cache metadata if it exists. (Safe for all processes)
        """
        metadata_file = cache_dir / dataset_name / split_name / "cache_metadata.json"
        if not metadata_file.exists():
            return None
        try:
            with open(metadata_file, "r") as fh:
                meta = json.load(fh)
            return meta
        except Exception as exc:
            print(
                f"⚠️  Warning: Failed to read cache metadata from {metadata_file}: {exc}")
            return None

    def _load_from_final_cache(
        self,
        cache_dir: Path,
        dataset_name: str,
        split_name: str,
    ) -> CutSet:
        """
        Load fully processed cuts from final cache. (Safe for all processes)
        """
        cache_path = cache_dir / dataset_name / split_name / "cuts_windowed.jsonl.gz"
        print(f"Loading from cache: {cache_path}")
        return CutSet.from_file(cache_path)  # Use lazy loading

    def _get_split_dataset_size_from_cache_or_compute(
        self,
        cut_sets: Dict[str, Dict[str, CutSet]],
        cache_dir: Path,
        dataset_name: str,
        split_name: str,
        label_type: LabelType,
        accelerator: Accelerator
    ) -> int:
        """
        Get dataset size for a specific split by reading cache metadata, and
        fall back to a direct (and potentially expensive) computation.
        """
        # Try to load metadata first
        meta = self._load_cache_metadata(cache_dir, dataset_name, split_name)
        if meta and isinstance(meta.get("dataset_size"), int):
            try:
                print(
                    f"✓ Loaded dataset_size from cache metadata for {dataset_name}/{split_name}: {meta['dataset_size']}")
            except Exception:
                pass
            return int(meta["dataset_size"])

        # Fall back to counting / computation.
        try:
            cuts = cut_sets[dataset_name][split_name]
        except KeyError:
            return 0

        size = 0
        if label_type == "ego":
            if accelerator.is_main_process:
                print(
                    f"→ Computing total dataset size for {dataset_name}/{split_name} (this may take a while)...")
                size = int(EgoCentricDiarizationDataset.get_total_dataset_size(
                    cuts, desc=f"Calculating total dataset size for {dataset_name}/{split_name}"))
                print(
                    f"✓ Computed dataset size for {dataset_name}/{split_name}: {size}")

                # Save metadata once so other processes can read it
                self._save_cache_metadata(cache_dir, dataset_name, split_name,
                                          dataset_size=size, num_cuts=len(cuts), label_type=label_type)

            # Barrier: ensure metadata file is written by main process
            accelerator.wait_for_everyone()

            # Non-main process: read metadata once and return value
            meta = self._load_cache_metadata(cache_dir, dataset_name, split_name)
            if meta and isinstance(meta['dataset_size'], int):
                size = int(meta['dataset_size'])
            else:
               raise RuntimeError(
                   f"Failed to load dataset_size from cache metadata for {dataset_name}/{split_name}")
        else:
            size = len(cuts)
        # Optionally save metadata to avoid future recompute

        if accelerator.is_main_process:
            self._save_cache_metadata(cache_dir, dataset_name, split_name,
                                      dataset_size=size, num_cuts=len(cuts), label_type=label_type)
        else:
            accelerator.wait_for_everyone()

        return size

    def _build_index_map(
        self,
        cuts: CutSet,
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Build the cut-speaker index map for ego-centric training.
        Returns list of (cut_id, speaker_id) tuples.
        """
        cut_speaker_map: List[Tuple[str, Optional[str]]] = []

        for cut in list(cuts):
            all_speakers_in_cut = sorted(
                set(s.speaker for s in cut.supervisions if s.speaker)
            )

            # Add entry for each speaker
            for target_spk_id in all_speakers_in_cut:
                cut_speaker_map.append((cut.id, target_spk_id))

            # Add entry for "no target speaker" case
            cut_speaker_map.append((cut.id, None))

        return cut_speaker_map

    def _apply_subsetting(
        self,
        cuts: CutSet,
        dataset_name: str,
        split_name: str,
        dataset_mapping: Optional[TrainingDatasetMap],
    ) -> CutSet:
        """
        Apply subsetting to cuts based on dataset mapping configuration. (Main process only)
        """
        if dataset_mapping is None or not dataset_mapping.splits:
            return cuts

        # Find matching split info in the mapping
        for split_info in dataset_mapping.splits:
            if (split_info.dataset_name == dataset_name and
                    split_info.split_name == split_name):
                subset_ratio = split_info.subset_ratio

                if 0 < subset_ratio < 1.0:
                    cuts = cuts.to_eager()  # Need eager to get len
                    num_cuts = int(len(cuts) * subset_ratio)
                    cuts = cuts.subset(first=num_cuts)
                    print(
                        f"  → Subsetting {dataset_name}/{split_name}: "
                        f"{subset_ratio:.2%} ({num_cuts} cuts)"
                    )
                break

        return cuts

    def _apply_windowing(
        self,
        cuts: CutSet,
        chunk_size: Optional[float],
        split_name: str,
    ) -> CutSet:
        """
        Apply windowing/chunking to cuts. (Main process only)
        Note: Labels should be computed AFTER windowing, not before.
        """
        if chunk_size is None or chunk_size <= 0:
            return cuts

        print(
            f"Applying windowing: {chunk_size}s windows for {split_name}")

        cuts = cuts.to_eager()  # Eager before windowing

        cuts = cuts.cut_into_windows(
            duration=chunk_size,
            hop=chunk_size / 2,
        ).pad(duration=chunk_size)

        cuts = cuts.to_eager()  # Eager after to get len

        print(f"Windowed cuts count: {len(cuts)}")
        return cuts

    def _load_precomputed_dataset(
        self,
        dataset: Any,
    ) -> Dict[str, CutSet]:
        """Load a dataset from precomputed manifests. (Main process only)"""

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

        global_config = getattr(dataset, "global_config", None)
        storage_path = Path(global_config.storage_path) if global_config and hasattr(
            global_config, 'storage_path') else None

        manifests = self._try_load_existing_manifests(
            manifest_dir, dataset.name, storage_path
        )

        if manifests is None:
            raise ValueError(
                f"Expected precomputed manifests for {dataset.name} in {manifest_dir}"
            )

        dataset_cut_sets = self._manifests_to_cutsets_dict(
            manifests, dataset.name
        )

        dataset_cut_sets = self._normalize_splits(
            dataset_cut_sets, dataset.name
        )

        dataset_cut_sets = self._process_features_for_dataset(
            dataset, dataset_cut_sets
        )

        print(
            f"✓ {dataset.name} ready (loaded from cached manifests)!\n")
        return dataset_cut_sets

    def load_datasets(
        self,
        datasets: List[Any],
        global_config: GlobalConfig,
        training_dataset_mapping: TrainingDatasetMap,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        cache_dir: Optional[Pathlike] = None,
        validation_dataset_mapping: Optional[TrainingDatasetMap] = None,
    ) -> Dict[str, Dict[str, CutSet]]:
        """
        Load datasets with smart distributed caching.

        PHASE 1: Main process checks cache. If invalid, it runs the full pipeline
                 (Download -> Manifest -> Features -> Windowing) and saves to disk.
        PHASE 3: All processes load the fully prepared manifests from disk.

        Args:
            training_dataset_mapping: Required. Specifies which splits to use for training.
            validation_dataset_mapping: Optional. Specifies which splits to use for validation.
        """
        params = LoadDatasetsParams(
            datasets=datasets,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # -------------------------------------------------------------------------
        # PHASE 1: PREPARATION (Main Process Only)
        # -------------------------------------------------------------------------

        print(f"\n{'='*80}")
        print(
            f"PHASE 1: Dataset Preparation (Main Process Only)")
        print(f"{'='*80}")

        for dataset in params.datasets:
            print(f"\nProcessing dataset: {dataset.name}")

            # Determine final cache directory
            final_cache_dir = None
            if cache_dir:
                final_cache_dir = Path(cache_dir)
            elif global_config and global_config.cache_dir:
                final_cache_dir = Path(global_config.cache_dir)

            if not final_cache_dir:
                raise ValueError(
                    f"No cache_dir specified for {dataset.name}. Provide one in load_datasets or global_config.")

            # Determine windowing
            data_loading = global_config.data_loading if global_config else None
            chunk_size = data_loading.chunk_size if data_loading else None

            # Check if this dataset is already fully cached
            # Use actual split names from training and validation mappings
            is_fully_cached = True
            splits_to_check = []
            found_splits = []

            # Collect all splits that should be cached for this dataset
            for split_info in training_dataset_mapping.splits:
                if split_info.dataset_name == dataset.name:
                    splits_to_check.append(split_info.split_name)

            if validation_dataset_mapping:
                for split_info in validation_dataset_mapping.splits:
                    if split_info.dataset_name == dataset.name:
                        splits_to_check.append(split_info.split_name)

            # If no splits found for this dataset, skip cache check
            if not splits_to_check:
                print(
                    f"→ {dataset.name}: No splits configured in mappings, will process all available splits")
                is_fully_cached = False
            else:
                # Check if all required splits are cached
                for split in splits_to_check:
                    if self._check_final_cache(final_cache_dir, dataset.name, split):
                        found_splits.append(split)

                # Cache is complete only if all required splits are found
                if len(found_splits) < len(splits_to_check):
                    is_fully_cached = False
                    missing = set(splits_to_check) - set(found_splits)
                    print(
                        f"→ {dataset.name}: Found {len(found_splits)}/{len(splits_to_check)} cached splits. Missing: {missing}")

            if is_fully_cached:
                print(
                    f"✓ {dataset.name}: Found all cached splits {found_splits} (Skipping processing)")
                continue

            print(
                f"→ {dataset.name}: Cache missing or incomplete. Processing from scratch...")

            # 1. Load/Compute base CutSets (Download -> Manifests -> Features)
            if getattr(dataset, "precomputed_only", False):
                dataset_cut_sets = self._load_precomputed_dataset(dataset)
            else:
                process_function, download_function = import_recipe(
                    dataset.name)
                dataset_cut_sets = self._process_single_dataset(
                    dataset, process_function, download_function
                )

            # 2. Determine label type from global config
            label_type: LabelType = "binary"
            if data_loading and hasattr(data_loading, 'label_type'):
                label_type = data_loading.label_type
            elif global_config and hasattr(global_config, 'label_type'):
                label_type = global_config.label_type

            # Get number of workers for label computation
            num_workers = 32  # Default
            if data_loading and hasattr(data_loading, 'num_workers'):
                num_workers = data_loading.num_workers

            # 3. Apply Subsetting, Windowing, Label Computation, and Save to Final Cache
            if chunk_size and chunk_size > 0:
                for split_name, cuts in dataset_cut_sets.items():
                    # Apply subsetting first (before windowing)
                    cuts_subsetted = self._apply_subsetting(
                        cuts, dataset.name, split_name, training_dataset_mapping
                    )
                    # Apply windowing BEFORE label computation
                    cuts_windowed = self._apply_windowing(
                        cuts_subsetted, chunk_size, split_name
                    )
                    # Labels generated on-the-fly in dataset, no pre-computation needed

                    # Compute dataset size (once) and save metadata with cache
                    num_cuts = len(cuts_windowed)
                    if label_type == "ego":
                        dataset_size = EgoCentricDiarizationDataset.get_total_dataset_size(
                            cuts_windowed, desc=f"Calculating total dataset size for {dataset.name}/{split_name}")
                    else:
                        dataset_size = num_cuts

                    # Save to final cache
                    self._save_to_final_cache(
                        cuts_windowed, final_cache_dir, dataset.name, split_name
                    )

                    # Save metadata about the cached split
                    try:
                        self._save_cache_metadata(
                            final_cache_dir,
                            dataset.name,
                            split_name,
                            dataset_size=dataset_size,
                            num_cuts=num_cuts,
                            label_type=label_type,
                        )
                    except Exception as exc:
                        print(
                            f"⚠️  Warning: Failed to write cache metadata for {dataset.name}/{split_name}: {exc}")
            else:
                print(
                    f"→ {dataset.name}: No chunk_size, applying subsetting, label computation, and saving to cache.")
                for split_name, cuts in dataset_cut_sets.items():
                    # Apply subsetting
                    cuts_subsetted = self._apply_subsetting(
                        cuts, dataset.name, split_name, training_dataset_mapping
                    )
                    # Labels generated on-the-fly in dataset, no pre-computation needed

                    num_cuts = len(cuts_subsetted)
                    if label_type == "ego":
                        dataset_size = EgoCentricDiarizationDataset.get_total_dataset_size(
                            cuts_subsetted, desc=f"Calculating total dataset size for {dataset.name}/{split_name}")
                    else:
                        dataset_size = num_cuts

                    self._save_to_final_cache(
                        cuts_subsetted, final_cache_dir, dataset.name, split_name
                    )

                    try:
                        self._save_cache_metadata(
                            final_cache_dir,
                            dataset.name,
                            split_name,
                            dataset_size=dataset_size,
                            num_cuts=num_cuts,
                            label_type=label_type,
                        )
                    except Exception as exc:
                        print(
                            f"⚠️  Warning: Failed to write cache metadata for {dataset.name}/{split_name}: {exc}")

            print(
                f"✓ {dataset.name} processing complete.")

            # Sync after each dataset to prevent timeout on large saves
            print(
                f"→ Syncing after completing {dataset.name}...")

        print(
            "Main process finished all preparation. Final sync...")
        print("All processes synced.")

        # -------------------------------------------------------------------------
        # PHASE 3: LOADING (All Processes)
        # -------------------------------------------------------------------------
        print(f"\n{'='*80}")
        print(
            f"PHASE 3: Loading Prepared Data (All Processes)")
        print(f"{'='*80}")

        all_cut_sets: Dict[str, Dict[str, CutSet]] = {}

        for dataset in params.datasets:
            final_cache_dir = Path(cache_dir) if cache_dir else Path(
                global_config.cache_dir)

            dataset_cuts = {}
            # Find all cached splits for this dataset by scanning the cache directory
            dataset_cache_dir = final_cache_dir / dataset.name
            if dataset_cache_dir.is_dir():
                for split_dir in dataset_cache_dir.iterdir():
                    if split_dir.is_dir():
                        split_name = split_dir.name
                        cache_file = split_dir / "cuts_windowed.jsonl.gz"
                        if cache_file.exists():
                            print(
                                f"Loading {dataset.name}/{split_name} from {cache_file}")
                            dataset_cuts[split_name] = CutSet.from_file(
                                cache_file)
                        else:
                            print(
                                f"⚠️  Warning: Found directory {split_dir} but no cuts_windowed.jsonl.gz inside")
            else:
                print(
                    f"⚠️  Warning: Cache directory not found: {dataset_cache_dir}")

            if not dataset_cuts:
                print(
                    f"⚠️  Warning: No cached splits loaded for {dataset.name}. Check cache_dir and permissions.")
            else:
                all_cut_sets[dataset.name] = dataset_cuts
                print(
                    f"✓ {dataset.name} loaded with {len(dataset_cuts)} splits: {list(dataset_cuts.keys())}")

        return all_cut_sets
