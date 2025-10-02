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
from typing import Any, Dict, List, Type, Union

import yaml
from yamlargparse import ArgumentParser

from .dataset_types import (
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
    FleursDownloadParams,
    FleursProcessParams,
    GigastDownloadParams,
    GigastProcessParams,
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
    "voxceleb": VoxcelebProcessParams,
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
    """Custom exception for dataset configuration errors"""

    pass


def validate_dataset_config(
    dataset_config: Dict[str, Any], global_config: Dict[str, Any] = None
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
    if "name" not in dataset_config:
        raise DatasetConfigError("Dataset configuration must include 'name' field")

    dataset_name = dataset_config["name"].lower()

    # Validate dataset name exists
    if dataset_name not in DATASET_DOWNLOAD_PARAMS_MAP:
        available_datasets = ", ".join(sorted(DATASET_DOWNLOAD_PARAMS_MAP.keys()))
        raise DatasetConfigError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available_datasets}"
        )

    # Get the appropriate dataclass types
    download_params_class = DATASET_DOWNLOAD_PARAMS_MAP[dataset_name]
    process_params_class = DATASET_PROCESS_PARAMS_MAP[dataset_name]

    # Apply global config defaults
    download_params_dict = dataset_config.get("download_params", {})
    process_params_dict = dataset_config.get("process_params", {})

    if global_config:
        # Apply global download defaults
        global_download = global_config.get("download_params", {})
        for key, value in global_download.items():
            if key not in download_params_dict:
                download_params_dict[key] = value

        # Apply global process defaults
        global_process = global_config.get("process_params", {})
        for key, value in global_process.items():
            if key not in process_params_dict:
                process_params_dict[key] = value

        # Apply global corpus_dir and output_dir defaults
        global_corpus_dir = global_config.get("corpus_dir")
        global_output_dir = global_config.get("output_dir")

        # Apply global corpus_dir to download target_dir
        if global_corpus_dir:
            # Set download target_dir to global corpus_dir
            if "target_dir" not in download_params_dict:
                download_params_dict["target_dir"] = global_corpus_dir

            # Note: corpus_dir for processing will be determined by the download function
            # We don't set it here as different datasets extract to different subdirectories

        if global_output_dir and "output_dir" not in process_params_dict:
            # Create dataset-specific output directory: global_output_dir/dataset_name
            process_params_dict["output_dir"] = f"{global_output_dir}/{dataset_name}"

    # Extract and validate download parameters
    try:
        download_params = download_params_class(**download_params_dict)
    except TypeError as e:
        raise DatasetConfigError(f"Invalid download parameters for {dataset_name}: {e}")

    # Extract and validate process parameters
    try:
        process_params = process_params_class(**process_params_dict)
    except TypeError as e:
        raise DatasetConfigError(f"Invalid process parameters for {dataset_name}: {e}")

    return DatasetConfig(
        name=dataset_name,
        download_params=download_params,
        process_params=process_params,
    )


def parse_dataset_configs(config_path: Union[str, Path]) -> List[DatasetConfig]:
    """
    Parse YAML configuration file and extract dataset configurations

    Args:
        config_path: Path to YAML configuration file

    Returns:
        List of validated DatasetConfig objects

    Raises:
        DatasetConfigError: If parsing or validation fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise DatasetConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise DatasetConfigError(f"Invalid YAML file: {e}")

    if not isinstance(config_data, dict):
        raise DatasetConfigError("Configuration file must contain a dictionary")

    if "datasets" not in config_data:
        raise DatasetConfigError("Configuration file must contain 'datasets' field")

    datasets_config = config_data["datasets"]
    if not isinstance(datasets_config, list):
        raise DatasetConfigError("'datasets' field must be a list")

    if not datasets_config:
        raise DatasetConfigError("'datasets' list cannot be empty")

    # Extract global configuration
    global_config = config_data.get("global_config", {})

    validated_configs = []
    for i, dataset_config in enumerate(datasets_config):
        try:
            validated_config = validate_dataset_config(dataset_config, global_config)
            validated_configs.append(validated_config)
        except DatasetConfigError as e:
            raise DatasetConfigError(f"Error in dataset configuration {i + 1}: {e}")

    return validated_configs


def datasets_manager_parser():
    """
    Parse command line arguments and return dataset configurations

    Returns:
        Tuple of (args, dataset_configs) where:
        - args: Parsed command line arguments
        - dataset_configs: List of validated DatasetConfig objects
    """
    parser = ArgumentParser(description="Datasets Manager")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    try:
        dataset_configs = parse_dataset_configs(args.config)
        return args, dataset_configs
    except DatasetConfigError as e:
        parser.error(str(e))


# Example usage
if __name__ == "__main__":
    args, dataset_configs = datasets_manager_parser()
    print("Parsed arguments:", args)
    print("\nDataset configurations:")
    for i, config in enumerate(dataset_configs):
        print(f"\nDataset {i + 1}: {config.name}")
        print(f"  Download params: {config.download_params}")
        print(f"  Process params: {config.process_params}")
