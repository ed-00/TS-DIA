"""
TS-DIA Datasets Package

This package provides a comprehensive dataset management system for the TS-DIA project,
supporting both typed and dictionary-based configuration approaches with global defaults.

Key Features:
- Global configuration system to eliminate repetition
- Automatic path construction for datasets and manifests
- Support for 50+ speech datasets via Lhotse integration
- Hybrid typed/dict parameter system for flexibility
- Dataset-specific manifest organization

Quick Start:
    ```python
    from datasets import DatasetManager, parse_dataset_configs

    # Load datasets from YAML config
    configs = parse_dataset_configs('configs/my_datasets.yml')
    cut_sets = DatasetManager.load_datasets(datasets=configs)
    ```

Configuration Example:
    ```yaml
    global_config:
      corpus_dir: ./data
      output_dir: ./manifests
      force_download: false

    datasets:
      - name: yesno
      - name: timit
        process_params:
          num_phones: 48
    ```

Classes:
    DatasetManager: Main class for loading and processing datasets
    DatasetConfig: Configuration for individual datasets
    LoadDatasetsParams: Parameters for dataset loading
    GlobalConfig: Global configuration defaults

Functions:
    parse_dataset_configs: Parse YAML configuration files
    list_available_datasets: List all supported datasets
    import_recipe: Import dataset-specific recipes
    select_recipe: Select appropriate recipe for dataset
"""

from . import recipes
from .dataset_types import (
    DatasetConfig,
    GlobalConfig,
    LoadDatasetsParams,
)
from .parse_args import (
    datasets_manager_parser,
    parse_dataset_configs,
    validate_dataset_config,
)

__all__ = [
    # Core classes
    "DatasetConfig",
    "LoadDatasetsParams",
    "GlobalConfig",
    # Configuration functions
    "parse_dataset_configs",
    "validate_dataset_config",
    "datasets_manager_parser",
    # Subpackages
    "recipes",
]

__version__ = "0.1.0"
__author__ = "TS-DIA Team"
