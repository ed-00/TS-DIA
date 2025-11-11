#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Configuration Parsing and Validation

This module provides functions to parse YAML configuration files and validate dataset
configurations with global defaults support. It handles the merging of global configuration
settings with dataset-specific parameters and ensures proper path construction.

Key Functions:
    parse_dataset_configs: Parse YAML file and return validated DatasetConfig objects
    validate_dataset_config: Validate and merge global config with dataset config
    datasets_manager_parser: Command-line argument parser for dataset management

Global Configuration Features:
- Automatic path construction: global_corpus_dir/dataset_name
- Dataset-specific manifest directories: global_output_dir/dataset_name
- Global download and processing defaults
- Override support for individual datasets

Example YAML Configuration:
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

Usage:
    ```python
    from datasets.parse_args import parse_dataset_configs

    # Parse configuration file
    configs = parse_dataset_configs('configs/my_datasets.yml')

    # Each config has global defaults applied
    for config in configs:
        print(f"Dataset: {config.name}")
        print(f"Download: {config.get_download_kwargs()}")
        print(f"Process: {config.get_process_kwargs()}")
    ```
"""

from pathlib import Path
from dataclasses import fields
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
from types import SimpleNamespace

from dataclasses_json import global_config
import yaml
from yamlargparse import ArgumentParser

from data_manager.dataset_types import (
    DataLoaderConfig,
    DataLoadingConfig,
    InputStrategyConfig,
)

from data_manager.dataset_types import (
    # Download params
    AdeptDownloadParams,
    # Process params
    AdeptProcessParams,
    Aishell3DownloadParams,
    Aishell3ProcessParams,
    Aishell4DownloadParams,
    Aishell4ProcessParams,
    AishellDownloadParams,
    AishellProcessParams,
    AliMeetingDownloadParams,
    AliMeetingProcessParams,
    AmiDownloadParams,
    AmiProcessParams,
    AtcosimDownloadParams,
    AtcosimProcessParams,
    AvaAvdDownloadParams,
    AvaAvdProcessParams,
    BakerZhDownloadParams,
    BakerZhProcessParams,
    BaseDownloadParams,
    BaseProcessParams,
    ButReverbDbDownloadParams,
    ButReverbDbProcessParams,
    BvccDownloadParams,
    BvccProcessParams,
    Chime6DownloadParams,
    Chime6ProcessParams,
    CmuArcticDownloadParams,
    CmuArcticProcessParams,
    CmuIndicDownloadParams,
    CmuIndicProcessParams,
    DailyTalkDownloadParams,
    DailyTalkProcessParams,
    # Core configs
    DatasetConfig,
    DipcoDownloadParams,
    DipcoProcessParams,
    Earnings21DownloadParams,
    Earnings21ProcessParams,
    Earnings22DownloadParams,
    Earnings22ProcessParams,
    EarsDownloadParams,
    EarsProcessParams,
    EdaccDownloadParams,
    EdaccProcessParams,
    Ego4dDownloadParams,
    Ego4dProcessParams,
    FeatureConfig,
    FleursDownloadParams,
    FleursProcessParams,
    GigastDownloadParams,
    GigastProcessParams,
    GlobalConfig,
    GridDownloadParams,
    GridProcessParams,
    HeroicoDownloadParams,
    HeroicoProcessParams,
    HifittsDownloadParams,
    HifittsProcessParams,
    HimiaDownloadParams,
    HimiaProcessParams,
    IcsiDownloadParams,
    IcsiProcessParams,
    LibricssDownloadParams,
    LibricssProcessParams,
    LibriheavyMixDownloadParams,
    LibriheavyMixProcessParams,
    LibrimixDownloadParams,
    LibrimixProcessParams,
    LibrispeechDownloadParams,
    LibrispeechProcessParams,
    LibrittsDownloadParams,
    LibrittsProcessParams,
    LibrittsrDownloadParams,
    LibrittsrProcessParams,
    LjspeechDownloadParams,
    LjspeechProcessParams,
    MagicdataDownloadParams,
    MagicdataProcessParams,
    MdccDownloadParams,
    MdccProcessParams,
    MedicalDownloadParams,
    MedicalProcessParams,
    MobvoihotwordsDownloadParams,
    MobvoihotwordsProcessParams,
    MswildDownloadParams,
    MswildProcessParams,
    MtedxDownloadParams,
    MtedxProcessParams,
    MusanDownloadParams,
    MusanProcessParams,
    NscProcessParams,
    PeoplesSpeechProcessParams,
    PrimewordsDownloadParams,
    PrimewordsProcessParams,
    ReazonspeechDownloadParams,
    ReazonspeechProcessParams,
    RirNoiseDownloadParams,
    RirNoiseProcessParams,
    SpeechcommandsDownloadParams,
    SpeechcommandsProcessParams,
    SpgispeechDownloadParams,
    SpgispeechProcessParams,
    StcmdsDownloadParams,
    StcmdsProcessParams,
    TedliumDownloadParams,
    TedliumProcessParams,
    Thchs30DownloadParams,
    Thchs30ProcessParams,
    ThisAmericanLifeDownloadParams,
    ThisAmericanLifeProcessParams,
    TimitDownloadParams,
    TimitProcessParams,
    UwbAtccDownloadParams,
    UwbAtccProcessParams,
    VctkDownloadParams,
    VctkProcessParams,
    Voxceleb1DownloadParams,
    Voxceleb2DownloadParams,
    VoxcelebProcessParams,
    VoxconverseDownloadParams,
    VoxconverseProcessParams,
    VoxpopuliDownloadParams,
    VoxpopuliProcessParams,
    XbmuAmdo31DownloadParams,
    XbmuAmdo31ProcessParams,
    YesnoDownloadParams,
    YesnoProcessParams,
)

# Dataset name to dataclass mapping
DATASET_DOWNLOAD_PARAMS_MAP: Dict[str, Type[BaseDownloadParams]] = {
    "adept": AdeptDownloadParams,
    "aishell": AishellDownloadParams,
    "aishell3": Aishell3DownloadParams,
    "aishell4": Aishell4DownloadParams,
    "ali_meeting": AliMeetingDownloadParams,
    "ami": AmiDownloadParams,
    "atcosim": AtcosimDownloadParams,
    "baker_zh": BakerZhDownloadParams,
    "but_reverb_db": ButReverbDbDownloadParams,
    "bvcc": BvccDownloadParams,
    "chime6": Chime6DownloadParams,
    "cmu_arctic": CmuArcticDownloadParams,
    "cmu_indic": CmuIndicDownloadParams,
    "daily_talk": DailyTalkDownloadParams,
    "dipco": DipcoDownloadParams,
    "earnings21": Earnings21DownloadParams,
    "earnings22": Earnings22DownloadParams,
    "ears": EarsDownloadParams,
    "ego4d": Ego4dDownloadParams,
    "edacc": EdaccDownloadParams,
    "fleurs": FleursDownloadParams,
    "gigast": GigastDownloadParams,
    "grid": GridDownloadParams,
    "heroico": HeroicoDownloadParams,
    "hifitts": HifittsDownloadParams,
    "himia": HimiaDownloadParams,
    "icsi": IcsiDownloadParams,
    "libricss": LibricssDownloadParams,
    "librimix": LibrimixDownloadParams,
    "libriheavy_mix": LibriheavyMixDownloadParams,
    "librispeech": LibrispeechDownloadParams,
    "libritts": LibrittsDownloadParams,
    "librittsr": LibrittsrDownloadParams,
    "ljspeech": LjspeechDownloadParams,
    "magicdata": MagicdataDownloadParams,
    "mdcc": MdccDownloadParams,
    "medical": MedicalDownloadParams,
    "mobvoihotwords": MobvoihotwordsDownloadParams,
    "mswild": MswildDownloadParams,
    "mtedx": MtedxDownloadParams,
    "voxconverse": VoxconverseDownloadParams,
    "ava_avd": AvaAvdDownloadParams,
    "musan": MusanDownloadParams,
    "primewords": PrimewordsDownloadParams,
    "spgispeech": SpgispeechDownloadParams,
    "stcmds": StcmdsDownloadParams,
    "tedlium": TedliumDownloadParams,
    "vctk": VctkDownloadParams,
    "voxceleb1": Voxceleb1DownloadParams,
    "voxceleb2": Voxceleb2DownloadParams,
    "reazonspeech": ReazonspeechDownloadParams,
    "rir_noise": RirNoiseDownloadParams,
    "speechcommands": SpeechcommandsDownloadParams,
    "thchs_30": Thchs30DownloadParams,
    "this_american_life": ThisAmericanLifeDownloadParams,
    "timit": TimitDownloadParams,
    "uwb_atcc": UwbAtccDownloadParams,
    "voxpopuli": VoxpopuliDownloadParams,
    "xbmu_amdo31": XbmuAmdo31DownloadParams,
    "yesno": YesnoDownloadParams,
}

DATASET_PROCESS_PARAMS_MAP: Dict[str, Type[BaseProcessParams]] = {
    "adept": AdeptProcessParams,
    "aishell": AishellProcessParams,
    "aishell3": Aishell3ProcessParams,
    "aishell4": Aishell4ProcessParams,
    "ali_meeting": AliMeetingProcessParams,
    "ami": AmiProcessParams,
    "atcosim": AtcosimProcessParams,
    "baker_zh": BakerZhProcessParams,
    "but_reverb_db": ButReverbDbProcessParams,
    "bvcc": BvccProcessParams,
    "chime6": Chime6ProcessParams,
    "cmu_arctic": CmuArcticProcessParams,
    "cmu_indic": CmuIndicProcessParams,
    "daily_talk": DailyTalkProcessParams,
    "dipco": DipcoProcessParams,
    "earnings21": Earnings21ProcessParams,
    "earnings22": Earnings22ProcessParams,
    "ears": EarsProcessParams,
    "ego4d": Ego4dProcessParams,
    "edacc": EdaccProcessParams,
    "fleurs": FleursProcessParams,
    "gigast": GigastProcessParams,
    "grid": GridProcessParams,
    "heroico": HeroicoProcessParams,
    "hifitts": HifittsProcessParams,
    "himia": HimiaProcessParams,
    "icsi": IcsiProcessParams,
    "libricss": LibricssProcessParams,
    "librimix": LibrimixProcessParams,
    "libriheavy_mix": LibriheavyMixProcessParams,
    "librispeech": LibrispeechProcessParams,
    "libritts": LibrittsProcessParams,
    "librittsr": LibrittsrProcessParams,
    "ljspeech": LjspeechProcessParams,
    "magicdata": MagicdataProcessParams,
    "mdcc": MdccProcessParams,
    "medical": MedicalProcessParams,
    "mobvoihotwords": MobvoihotwordsProcessParams,
    "mswild": MswildProcessParams,
    "mtedx": MtedxProcessParams,
    "voxconverse": VoxconverseProcessParams,
    "ava_avd": AvaAvdProcessParams,
    "musan": MusanProcessParams,
    "nsc": NscProcessParams,
    "peoples_speech": PeoplesSpeechProcessParams,
    "primewords": PrimewordsProcessParams,
    "spgispeech": SpgispeechProcessParams,
    "stcmds": StcmdsProcessParams,
    "tedlium": TedliumProcessParams,
    "vctk": VctkProcessParams,
    "voxceleb1": VoxcelebProcessParams,
    "voxceleb2": VoxcelebProcessParams,
    "reazonspeech": ReazonspeechProcessParams,
    "rir_noise": RirNoiseProcessParams,
    "speechcommands": SpeechcommandsProcessParams,
    "thchs_30": Thchs30ProcessParams,
    "this_american_life": ThisAmericanLifeProcessParams,
    "timit": TimitProcessParams,
    "uwb_atcc": UwbAtccProcessParams,
    "voxpopuli": VoxpopuliProcessParams,
    "xbmu_amdo31": XbmuAmdo31ProcessParams,
    "yesno": YesnoProcessParams,
}


class DatasetConfigError(Exception):
    """Custom exception for dataset configuration errors."""


def _ensure_str_mapping(value: Any, context: str) -> Dict[str, Any]:
    """Return a shallow copy of value if it is a mapping with string keys."""

    if not isinstance(value, Mapping):
        raise DatasetConfigError(f"{context} must be a mapping")

    mapping_value = cast(Mapping[Any, Any], value)
    result: Dict[str, Any] = {}
    for key, item in mapping_value.items():
        if not isinstance(key, str):
            raise DatasetConfigError(f"{context} keys must be strings")
        result[key] = item
    return result


def _optional_str_mapping(value: Any, context: str) -> Dict[str, Any]:
    """Return a mapping if provided, otherwise an empty dict."""

    if value is None:
        return {}
    return _ensure_str_mapping(value, context)


def _filter_dataclass_kwargs(dataclass_type: Type[Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter mapping keys so that only dataclass fields remain."""

    valid_fields = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in data.items() if key in valid_fields}


def validate_dataset_config(
    dataset_config: Mapping[str, Any],
    global_config: Optional[Mapping[str, Any]] = None,
) -> DatasetConfig:
    """
    Validate and convert a dataset configuration dictionary to a DatasetConfig object

    Args:
        dataset_config: Dictionary containing dataset configuration
        global_config: Global configuration dictionary with defaults

    Returns:
        DatasetConfig object with validated parameters

    Raises:
        DatasetConfigError: If validation fails
    """
    dataset_config_dict = _ensure_str_mapping(
        dataset_config, "dataset configuration")

    dataset_name_raw = dataset_config_dict.get("name")
    if not isinstance(dataset_name_raw, str) or not dataset_name_raw.strip():
        raise DatasetConfigError(
            "Dataset configuration must include 'name' field")

    dataset_name = dataset_name_raw.lower()

    global_config_dict: Optional[Dict[str, Any]] = None
    if global_config is not None:
        global_config_dict = _ensure_str_mapping(
            global_config, "global configuration")

    manifest_dir_override = dataset_config_dict.get("manifest_dir")
    manifest_dir: Optional[Path] = None
    if isinstance(manifest_dir_override, str) and manifest_dir_override:
        manifest_dir = Path(manifest_dir_override)
    elif global_config_dict is not None:
        global_output_dir = global_config_dict.get("output_dir")
        if isinstance(global_output_dir, str) and global_output_dir:
            manifest_dir = Path(global_output_dir) / dataset_name

    # Validate dataset name exists or manifests are already prepared
    if dataset_name not in DATASET_DOWNLOAD_PARAMS_MAP:
        if manifest_dir is not None and manifest_dir.exists():
            precomputed_config = DatasetConfig(
                name=dataset_name,
                download_params=BaseDownloadParams(),
                process_params=BaseProcessParams(output_dir=str(manifest_dir)),
                precomputed_only=True,
                precomputed_manifest_dir=str(manifest_dir),
            )
            if global_config_dict is not None:
                precomputed_config.apply_global_config(global_config_dict)
            return precomputed_config

        available_datasets = ", ".join(
            sorted(DATASET_DOWNLOAD_PARAMS_MAP.keys()))
        raise DatasetConfigError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available_datasets}"
        )

    # Get the appropriate dataclass types
    download_params_class = DATASET_DOWNLOAD_PARAMS_MAP[dataset_name]
    process_params_class = DATASET_PROCESS_PARAMS_MAP[dataset_name]

    # Apply global config defaults
    download_params_dict = _optional_str_mapping(
        dataset_config_dict.get("download_params"),
        f"download_params for '{dataset_name}'",
    )
    process_params_dict = _optional_str_mapping(
        dataset_config_dict.get("process_params"),
        f"process_params for '{dataset_name}'",
    )

    if global_config_dict is not None:

        global_download = _optional_str_mapping(
            global_config_dict.get("download_params"),
            "global_config.download_params",
        )
        for key, value in global_download.items():
            download_params_dict.setdefault(key, value)

        global_process = _optional_str_mapping(
            global_config_dict.get("process_params"),
            "global_config.process_params",
        )
        for key, value in global_process.items():
            process_params_dict.setdefault(key, value)

        global_corpus_dir = global_config_dict.get("corpus_dir")
        global_output_dir = global_config_dict.get("output_dir")

        # Apply global corpus_dir to download target_dir
        if isinstance(global_corpus_dir, str) and global_corpus_dir:
            download_params_dict.setdefault("target_dir", global_corpus_dir)

        if isinstance(global_output_dir, str) and global_output_dir:
            process_params_dict.setdefault(
                "output_dir", f"{global_output_dir}/{dataset_name}"
            )

        # Pass sampling_rate from global_config to process_params for audio extraction
        # This ensures audio is resampled to match feature extraction requirements
        global_sampling_rate = global_config_dict.get("sampling_rate")
        if (
            isinstance(global_sampling_rate, (int, float))
            and "sampling_rate" not in process_params_dict
        ):
            process_params_dict["sampling_rate"] = global_sampling_rate

    # Extract and validate download parameters
    try:
        download_params = download_params_class(**download_params_dict)
    except TypeError as e:
        raise DatasetConfigError(
            f"Invalid download parameters for {dataset_name}: {e}"
        ) from e

    # Extract and validate process parameters
    try:
        process_params = process_params_class(**process_params_dict)
    except TypeError as e:
        raise DatasetConfigError(
            f"Invalid process parameters for {dataset_name}: {e}"
        ) from e

    dataset_obj = DatasetConfig(
        name=dataset_name,
        download_params=download_params,
        process_params=process_params,
    )
    if global_config_dict is not None:
        dataset_obj.apply_global_config(global_config_dict)
    return dataset_obj


def parse_dataset_configs(config_path: Union[str, Path]) -> Tuple[List[DatasetConfig], GlobalConfig]:
    """Parse a YAML configuration file and return validated dataset configs."""

    path_obj = Path(config_path)
    if not path_obj.exists():
        raise DatasetConfigError(f"Configuration file not found: {path_obj}")

    try:
        with open(path_obj, "r", encoding="utf-8") as config_file:
            config_data_raw = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise DatasetConfigError(f"Invalid YAML file: {exc}") from exc

    if not isinstance(config_data_raw, Mapping):
        raise DatasetConfigError(
            "Configuration file must contain a dictionary")

    config_data = _ensure_str_mapping(config_data_raw, "root configuration")

    datasets_section = config_data.get("datasets")
    if datasets_section is None:
        raise DatasetConfigError(
            "Configuration file must contain 'datasets' field")
    if not isinstance(datasets_section, list):
        raise DatasetConfigError("'datasets' field must be a list")
    if not datasets_section:
        raise DatasetConfigError("'datasets' list cannot be empty")

    datasets_config: List[Dict[str, Any]] = []
    datasets_section_list = cast(List[Any], datasets_section)
    for index, dataset_entry in enumerate(datasets_section_list):
        if not isinstance(dataset_entry, Mapping):
            raise DatasetConfigError(
                f"Dataset configuration at index {index} must be a mapping"
            )
        datasets_config.append(
            _ensure_str_mapping(dataset_entry, f"datasets[{index}]")
        )

    global_config_section = config_data.get("global_config")

    if global_config_section is None:
        raise DatasetConfigError(
            "Missing 'global_config' section in configuration. "
            "Please provide at least: corpus_dir, output_dir, and feature extraction settings."
        )
    global_config_dict = _ensure_str_mapping(
        global_config_section, "global_config")

    features_section = global_config_dict.get("features")
    if features_section is None:
        raise DatasetConfigError(
            "Missing 'features' configuration in global_config")
    features_dict = _ensure_str_mapping(
        features_section, "global_config.features")
    try:
        features = FeatureConfig(**features_dict)
    except TypeError as exc:
        # Get the field names expected by FeatureConfig for better error reporting
        expected_fields = [field.name for field in fields(FeatureConfig)]
        provided_fields = list(features_dict.keys())
        
        # Identify missing required fields and unknown fields
        missing_fields = [f for f in expected_fields if f not in provided_fields 
                         and getattr(fields(FeatureConfig), 'default', None) is None]
        unknown_fields = [f for f in provided_fields if f not in expected_fields]
        
        error_parts = [f"Invalid feature configuration: {exc}"]
        if missing_fields:
            error_parts.append(f"Missing required fields: {', '.join(missing_fields)}")
        if unknown_fields:
            error_parts.append(f"Unknown fields: {', '.join(unknown_fields)}")
        error_parts.append(f"Expected fields: {', '.join(expected_fields)}")
        error_parts.append(f"Provided fields: {', '.join(provided_fields)}")
        
        raise DatasetConfigError("\n".join(error_parts)) from exc

    data_loading_dict = _optional_str_mapping(
        global_config_dict.get("data_loading"),
        "global_config.data_loading",
    )

    dataloader_dict = _optional_str_mapping(
        data_loading_dict.get("dataloader"),
        "global_config.data_loading.dataloader",
    )

    dataloader_cfg = DataLoaderConfig(
        **_filter_dataclass_kwargs(DataLoaderConfig, dataloader_dict)
    )

    strategy_raw = data_loading_dict.get("strategy", "precomputed_features")
    if not isinstance(strategy_raw, str):
        raise DatasetConfigError(
            "global_config.data_loading.strategy must be a string when provided"
        )
    allowed_strategies = {
        "precomputed_features",
        "on_the_fly_features",
        "audio_samples",
    }
    if strategy_raw not in allowed_strategies:
        raise DatasetConfigError(
            "global_config.data_loading.strategy must be one of: "
            "precomputed_features, on_the_fly_features, audio_samples"
        )
    strategy_value = cast(
        Literal["precomputed_features",
                "on_the_fly_features", "audio_samples"],
        strategy_raw,
    )

    # Extract frame_stack from data_loading config
    frame_stack_value = data_loading_dict.get("frame_stack", 1)
    if not isinstance(frame_stack_value, int):
        raise DatasetConfigError(
            "global_config.data_loading.frame_stack must be an integer when provided"
        )

    # Extract subsampling from data_loading config
    subsampling_value = data_loading_dict.get("subsampling", 1)
    if not isinstance(subsampling_value, int):
        raise DatasetConfigError(
            "global_config.data_loading.subsampling must be an integer when provided"
        )

    # Extract chunk_size from data_loading config
    chunk_size_value = data_loading_dict.get("chunk_size")
    if chunk_size_value is not None and not isinstance(chunk_size_value, (int, float)):
        raise DatasetConfigError(
            "global_config.data_loading.chunk_size must be a number when provided"
        )

    # Extract context_size from data_loading config
    context_size_value = data_loading_dict.get("context_size", 7)
    if not isinstance(context_size_value, int):
        raise DatasetConfigError(
            "global_config.data_loading.context_size must be an integer when provided"
        )

    # Extract min_enroll_len from data_loading config
    min_enroll_len_value = data_loading_dict.get("min_enroll_len", 1.0)
    if not isinstance(min_enroll_len_value, (int, float)):
        raise DatasetConfigError(
            "global_config.data_loading.min_enroll_len must be a number when provided"
        )

    # Extract max_enroll_len from data_loading config
    max_enroll_len_value = data_loading_dict.get("max_enroll_len", 5.0)
    if not isinstance(max_enroll_len_value, (int, float)):
        raise DatasetConfigError(
            "global_config.data_loading.max_enroll_len must be a number when provided"
        )

    data_loading_cfg = DataLoadingConfig(
        strategy=strategy_value,
        frame_stack=frame_stack_value,
        subsampling=subsampling_value,
        chunk_size=chunk_size_value,
        context_size=context_size_value,
        min_enroll_len=min_enroll_len_value,
        max_enroll_len=max_enroll_len_value,

        dataloader=dataloader_cfg,
    )

    corpus_dir_value = global_config_dict.get("corpus_dir", "./data")
    if not isinstance(corpus_dir_value, str):
        raise DatasetConfigError("global_config.corpus_dir must be a string")
    output_dir_value = global_config_dict.get("output_dir", "./manifests")
    if not isinstance(output_dir_value, str):
        raise DatasetConfigError("global_config.output_dir must be a string")
    force_download_value = global_config_dict.get("force_download", False)
    if not isinstance(force_download_value, bool):
        raise DatasetConfigError(
            "global_config.force_download must be a boolean")
    storage_path_value = global_config_dict.get("storage_path")
    if storage_path_value is not None and not isinstance(storage_path_value, str):
        raise DatasetConfigError(
            "global_config.storage_path must be a string when provided"
        )
    random_seed_value = global_config_dict.get("random_seed", 42)
    if not isinstance(random_seed_value, int):
        raise DatasetConfigError(
            "global_config.random_seed must be an integer")

    try:
        global_config_obj = GlobalConfig(
            corpus_dir=corpus_dir_value,
            output_dir=output_dir_value,
            force_download=force_download_value,
            storage_path=storage_path_value,
            features=features,
            data_loading=data_loading_cfg,
            random_seed=random_seed_value,
        )
    except TypeError as exc:
        raise DatasetConfigError(
            f"Invalid global configuration: {exc}") from exc

    validated_configs: List[DatasetConfig] = []
    for index, dataset_entry in enumerate(datasets_config):
        try:
            validated_config = validate_dataset_config(
                dataset_entry, global_config_dict
            )

            validated_config.global_config = global_config_obj
            validated_configs.append(validated_config)
        except DatasetConfigError as exc:
            raise DatasetConfigError(
                f"Error in dataset configuration {index + 1}: {exc}"
            ) from exc

    return validated_configs, global_config_obj


def datasets_manager_parser() -> Tuple[List[DatasetConfig], GlobalConfig]:
    """CLI parser for dataset management commands."""

    parser = ArgumentParser(description="Datasets Manager")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    # type: ignore[call-overload]
    args = parser.parse_args(namespace=SimpleNamespace())

    try:
        dataset_configs, global_config = parse_dataset_configs(args.config)
    except DatasetConfigError as exc:
        parser.error(str(exc))
        raise SystemExit(2) from exc

    return dataset_configs,  global_config


# Example usage
if __name__ == "__main__":
    def _main() -> None:
        dataset_configs, global_config = datasets_manager_parser()

        print("\nDataset configurations:")
        for index, config in enumerate(dataset_configs, start=1):
            print(f"\nDataset {index}: {config.name}")
            print(f"  Download params: {config.download_params}")
            print(f"  Process params: {config.process_params}")

    _main()
