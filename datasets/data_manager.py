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

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lhotse as lh
from lhotse import CutSet, RecordingSet, SupervisionSet

from . import recipes
from .dataset_types import LoadDatasetsParams
from .parse_args import datasets_manager_parser


def __is_custom_recipe(dataset_name: str) -> bool:
    """
    Check if a custom recipe exists for the dataset.

    Args:
        dataset_name: Name of the dataset to check

    Returns:
        True if custom recipe exists, False otherwise
    """
    return dataset_name in recipes.__all__


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
    def load_datasets(**kwargs) -> List[CutSet]:
        """
        Load datasets and convert all manifest formats to CutSets for diarization tasks.

        This method handles the complete pipeline from dataset configuration to CutSet generation:
        1. Downloads datasets if needed
        2. Processes datasets to generate manifests
        3. Converts manifests to Lhotse CutSets
        4. Returns list of CutSets ready for diarization tasks

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
            List[CutSet]: List of CutSet objects, one per dataset split

        Raises:
            ValueError: If dataset has no download or process function
            ValueError: If manifest conversion fails

        Example:
            ```python
            from datasets import DatasetManager, parse_dataset_configs

            # Load from configuration file
            configs = parse_dataset_configs('configs/my_datasets.yml')
            cut_sets = DatasetManager.load_datasets(datasets=configs)

            # Process each CutSet
            for cut_set in cut_sets:
                print(f"CutSet: {cut_set.describe()}")
            ```
        """
        params = LoadDatasetsParams(**kwargs)
        recipes = [
            (import_recipe(dataset.name), dataset) for dataset in params.datasets
        ]
        cut_sets = []

        for recipe in recipes:
            (process_function, download_function), dataset = recipe
            if not download_function and not process_function:
                raise ValueError(
                    f"Dataset {dataset.name} has no download or process function"
                )

            # Download dataset if download function exists
            corpus_path = None
            if download_function:
                corpus_path = download_function(**dataset.get_download_kwargs())

            # Process dataset to get manifests
            if process_function:
                # Use the actual corpus path returned by download function if available
                process_kwargs = dataset.get_process_kwargs()
                if corpus_path and "corpus_dir" not in process_kwargs:
                    process_kwargs["corpus_dir"] = corpus_path
                elif corpus_path and "corpus_dir" in process_kwargs:
                    # Override with actual path if download function provided one
                    process_kwargs["corpus_dir"] = corpus_path

                manifests = process_function(**process_kwargs)

                # Convert manifests to CutSet(s)
                dataset_cut_sets = DatasetManager._manifests_to_cutsets(
                    manifests, dataset.name
                )
                cut_sets.extend(dataset_cut_sets)

        return cut_sets

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
