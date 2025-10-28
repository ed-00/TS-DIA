# Datasets Module

The `datasets` module provides a comprehensive dataset management system for the TS-DIA project, supporting both typed and dictionary-based configuration approaches with global defaults.

## Quick Links

- **[📖 Complete Parameters Guide](PARAMETERS_GUIDE.md)** - Comprehensive reference for all dataset parameters and feature extraction
- **[📁 Ready-to-Use YAML Configs](../../configs/datasets/)** - Individual configuration files for 50+ datasets
- **[📚 Dataset Recipes](recipes/)** - Detailed documentation for each dataset

## Overview

This module offers a unified interface for downloading, processing, and managing speech datasets for diarization tasks. It integrates with Lhotse to support 50+ speech datasets and provides a flexible configuration system that eliminates repetition through global defaults.

## Key Features

- **Global Configuration System**: Set defaults once for all datasets
- **Automatic Path Construction**: Datasets and manifests organized automatically
- **Dataset-Specific Directories**: Each dataset gets its own manifest directory
- **Hybrid Parameter System**: Support for both typed and dictionary parameters
- **Lhotse Integration**: Access to 50+ speech datasets
- **Flexible Override Support**: Individual datasets can override global settings

## Quick Start

```python
from datasets import DatasetManager, parse_dataset_configs

# Load datasets from YAML configuration
configs = parse_dataset_configs('configs/my_datasets.yml')
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Process the CutSets
for cut_set in cut_sets:
    cut_set.describe()
```

## Configuration

### YAML Configuration Format

```yaml
# Global configuration (applied to all datasets)
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

# Dataset list (minimal configuration)
datasets:
  - name: yesno
  - name: timit
    process_params:
      num_phones: 48
  - name: librispeech
    download_params:
      dataset_parts: "mini_librispeech"
    process_params:
      normalize_text: "lower"

# Other parameters
other_params:
  num_workers: 4
  pin_memory: true
  validation_split: 0.1
  test_split: 0.1
```

### Global Configuration

The `global_config` section defines defaults for all datasets:

- **`corpus_dir`**: Base directory for dataset downloads (default: `./data`)
- **`output_dir`**: Base directory for manifest outputs (default: `./manifests`)
- **`force_download`**: Whether to force re-download (default: `false`)

### Dataset Configuration

Each dataset in the `datasets` list needs only:

- **`name`**: Dataset name (required)
- **`download_params`**: Download-specific parameters (optional)
- **`process_params`**: Processing-specific parameters (optional)

## Directory Structure

The system automatically creates organized directory structures:

```bash
./data/                          # Global corpus directory
├── waves_yesno/                 # Dataset-specific subdirectories
├── timit/
└── librispeech/

./manifests/                     # Global manifest directory
├── yesno/                       # Dataset-specific manifest directories
│   ├── yesno_recordings_train.jsonl.gz
│   ├── yesno_recordings_test.jsonl.gz
│   ├── yesno_supervisions_train.jsonl.gz
│   └── yesno_supervisions_test.jsonl.gz
├── timit/
└── librispeech/
````

## API Reference

### Core Classes

#### `DatasetManager`

Main class for loading and processing datasets.

```python
class DatasetManager:
    @staticmethod
    def load_datasets(**kwargs) -> List[CutSet]:
        """Load datasets and convert to CutSets for diarization tasks."""
```

#### `DatasetConfig`

Configuration for individual datasets.

```python
@dataclass
class DatasetConfig:
    name: str
    download_params: Union[BaseDownloadParams, Dict[str, Any]]
    process_params: Union[BaseProcessParams, Dict[str, Any]]
    
    def apply_global_config(self, global_config: Union[GlobalConfig, Dict[str, Any]]) -> None:
        """Apply global configuration to this dataset config."""
```

#### `GlobalConfig`

Global configuration defaults.

```python
@dataclass
class GlobalConfig:
    corpus_dir: str = "./data"
    output_dir: str = "./manifests"
    force_download: bool = False
```

### Configuration Functions

#### `parse_dataset_configs(config_path: Union[str, Path]) -> List[DatasetConfig]`

Parse YAML configuration file and return validated DatasetConfig objects.

```python
configs = parse_dataset_configs('configs/my_datasets.yml')
```

#### `validate_dataset_config(dataset_config: Dict[str, Any], global_config: Dict[str, Any] = None) -> DatasetConfig`

Validate and merge global config with dataset config.

### Utility Functions

#### `list_available_datasets() -> set[str]`

List all available datasets from both Lhotse and custom recipes.

```python
available = list_available_datasets()
print(f"Available datasets: {available}")
```

#### `import_recipe(dataset_name: str) -> Tuple[Callable, Optional[Callable]]`

Import dataset-specific download and process functions.

```python
process_func, download_func = import_recipe('timit')
```

## Supported Datasets

The system supports 50+ speech datasets via Lhotse integration, including:

- **TIMIT**: Phone recognition dataset
- **LibriSpeech**: Large-scale English speech recognition
- **VoxCeleb**: Speaker identification and verification
- **AMI**: Meeting recordings
- **ICSI**: Meeting recordings
- **CHIME6**: Multi-channel speech enhancement
- **YesNo**: Binary classification dataset
- **Ego4D**: First-person video dataset with audio diarization support
- And many more...

### Ego4D Dataset

The Ego4D dataset is a large-scale egocentric video dataset that includes audio-visual data from first-person perspectives. This recipe focuses on extracting and processing audio data for speaker diarization tasks.

#### Features
- **Audio Extraction**: Automatic extraction of audio from video files
- **Voice Activity Processing**: Processing of voice activity annotations
- **Dataset Parts Injection**: Proper injection of dataset parts into ego4d CLI command
- **Environment Variable Support**: Secure access key management with python-dotenv

#### Usage
```yaml
# Configuration example
datasets:
  - name: ego4d
    download_params:
      access_key: null  # Load from EGO4D_ACCESS_KEY environment variable
      dataset_parts: ["clips", "annotations"]  # Injected into ego4d CLI
      install_cli: true
      timeout: 3600
    process_params:
      extract_audio: true
      audio_sample_rate: 16000
      min_segment_duration: 0.5
      max_segment_duration: 30.0
```

#### Environment Setup
```bash
# Set your AWS credentials for Ego4D access
export AWS_ACCESS_KEY_ID="your_aws_access_key_id_here"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_access_key_here"

# Or create .env file
echo "AWS_ACCESS_KEY_ID=your_aws_access_key_id_here" > .env
echo "AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here" >> .env
```

#### Command Execution
The recipe executes the following ego4d CLI command:
```bash
ego4d --output_directory=./data --datasets=clips annotations --yes
```

## Command Line Usage

```bash
# Process datasets from configuration file
python -m datasets.data_manager --config configs/my_datasets.yml

# List available datasets
python -c "from datasets import list_available_datasets; print(list_available_datasets())"
```

## Examples

### Basic Usage

```python
from datasets import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/basic_datasets.yml')

# Process datasets
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use CutSets for diarization
for cut_set in cut_sets:
    cut_set.describe()
```

### Advanced Configuration

```python
from datasets import DatasetConfig, GlobalConfig, DatasetManager

# Create global configuration
global_config = GlobalConfig(
    corpus_dir="./my_data",
    output_dir="./my_manifests",
    force_download=True
)

# Create dataset configurations
datasets = [
    DatasetConfig(
        name="timit",
        process_params={"num_phones": 48}
    ),
    DatasetConfig(
        name="yesno",
        download_params={"force_download": False}  # Override global
    )
]

# Apply global config
for dataset in datasets:
    dataset.apply_global_config(global_config)

# Load datasets
cut_sets = DatasetManager.load_datasets(datasets=datasets)
```

### Custom Recipe Integration

```python
from datasets import import_recipe

# Import specific dataset functions
process_func, download_func = import_recipe('timit')

# Download dataset
corpus_path = download_func(target_dir='./data', force_download=True)

# Process dataset
manifests = process_func(
    corpus_dir=corpus_path,
    output_dir='./manifests/timit',
    num_phones=48
)
```

## Error Handling

The module provides comprehensive error handling:

- **`DatasetConfigError`**: Configuration validation errors
- **`ValueError`**: Dataset not found or processing errors
- **`FileNotFoundError`**: Missing configuration files

## Performance Considerations

- **Parallel Processing**: Supports multi-worker dataset processing
- **Memory Management**: Efficient CutSet generation and management
- **Caching**: Reuses downloaded datasets when possible
- **Lazy Loading**: Datasets loaded only when needed

## Contributing

To add support for new datasets:

1. Add dataset-specific parameter classes in `dataset_types.py`
2. Create custom recipes in `datasets/recipes/`
3. Update the parameter mapping dictionaries
4. Add tests and documentation

## License

Copyright (c) 2025 Abed Hameed. Licensed under Apache 2.0 license.
