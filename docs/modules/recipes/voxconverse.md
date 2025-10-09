# VoxConverse Speaker Diarization Recipe

## Overview

VoxConverse is a speaker diarization dataset containing audio extracted from YouTube videos with speaker diarization annotations in RTTM format. It's designed for evaluating speaker diarization systems on multi-speaker conversational audio.

## Features

- **RTTM Annotations**: Standard RTTM format for diarization evaluation
- **Multi-speaker Conversations**: Real-world conversational audio
- **Dev and Test Splits**: Separate development and test sets
- **Automatic Download**: Direct download from VoxConverse repository
- **Lhotse Integration**: Automatic conversion to Lhotse manifests

## Dataset Information

- **Name**: `voxconverse`
- **Type**: Speaker diarization
- **License**: Creative Commons (check official website for details)
- **Homepage**: https://www.robots.ox.ac.uk/~vgg/data/voxconverse/
- **Paper**: https://arxiv.org/abs/2005.12764
- **Splits**: Development, Test

## Prerequisites

### System Dependencies
- None (pure Python download and processing)

### Python Dependencies
- `lhotse` (for manifest generation)
- `tqdm` (for progress bars)

## Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str/Path | **Required** | Directory to download dataset to |
| `force_download` | bool | `False` | Force re-download even if already present |
| `download_dev` | bool | `True` | Download development split |
| `download_test` | bool | `True` | Download test split |

## Process Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str/Path | **Required** | Path to downloaded VoxConverse dataset |
| `output_dir` | str/Path | `None` | Path to output manifests directory |
| `splits` | Dict[str, str] | `{"dev": "dev", "test": "test"}` | Split name to directory mapping |

## Configuration Example

### Minimal Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: voxconverse
```

### Full Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: voxconverse
    download_params:
      force_download: false
      download_dev: true
      download_test: true
    process_params:
      splits:
        dev: "dev"
        test: "test"
```

### Development Only Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: voxconverse
    download_params:
      download_dev: true
      download_test: false
    process_params:
      splits:
        dev: "dev"
```

## Output Structure

```
./data/voxconverse/                  # Downloaded dataset
├── wavs/                            # Audio files
│   ├── dev/                         # Development split
│   │   └── audio/
│   │       ├── abjxc.wav
│   │       ├── abqrz.wav
│   │       └── ...
│   └── test/                        # Test split
│       └── audio/
│           ├── aecdo.wav
│           └── ...
├── rttms/                           # RTTM annotations
│   ├── dev/
│   │   ├── abjxc.rttm
│   │   ├── abqrz.rttm
│   │   └── ...
│   └── test/
│       ├── aecdo.rttm
│       └── ...
└── .completed                       # Download completion marker

./manifests/voxconverse/             # Generated manifests
├── voxconverse_recordings_dev.jsonl.gz
├── voxconverse_supervisions_dev.jsonl.gz
├── voxconverse_recordings_test.jsonl.gz
└── voxconverse_supervisions_test.jsonl.gz
```

## Usage Example

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/voxconverse_config.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use for diarization
for cut_set in cut_sets:
    print(f"Loaded {len(cut_set)} cuts")
    cut_set.describe()
```

### Direct Recipe Usage

```python
from data_manager.recipes.voxconverse import download_voxconverse, prepare_voxconverse

# Download
corpus_dir = download_voxconverse(
    target_dir="./data",
    download_dev=True,
    download_test=True
)

# Process
manifests = prepare_voxconverse(
    corpus_dir=corpus_dir,
    output_dir="./manifests/voxconverse"
)

# Access manifests
dev_recordings = manifests["dev"]["recordings"]
dev_supervisions = manifests["dev"]["supervisions"]
```

### Command Line

```bash
# Process VoxConverse dataset
python -m data_manager.data_manager --config configs/voxconverse_config.yml
```

## RTTM Format

VoxConverse uses standard RTTM (Rich Transcription Time Marked) format:

```rttm
SPEAKER file_id 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
```

Example:

```rttm
SPEAKER abjxc 1 0.50 1.20 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER abjxc 1 1.80 2.50 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER abjxc 1 4.50 1.00 <NA> <NA> SPEAKER_00 <NA> <NA>
```

Fields:

- **file_id**: Recording identifier
- **start_time**: Segment start in seconds
- **duration**: Segment duration in seconds
- **speaker_id**: Speaker label

## Dataset Statistics

### Development Set

- **Audio Files**: ~216 files
- **Total Duration**: ~15 hours
- **Speakers**: Multiple speakers per recording
- **Average Duration**: ~4 minutes per file

### Test Set  

- **Audio Files**: ~232 files
- **Total Duration**: ~17 hours
- **Speakers**: Multiple speakers per recording
- **Average Duration**: ~4.5 minutes per file

## Evaluation

VoxConverse is commonly evaluated using:

- **Diarization Error Rate (DER)**: Primary metric
- **Jaccard Error Rate (JER)**: Alternative metric
- **Collar**: Typically 0.25 seconds for segment boundaries

## Troubleshooting

### Missing Audio Files

```bash
WARNING: Audio file not found: .../voxconverse/wavs/dev/audio/xxx.wav
```

**Solution**: Ensure `download_dev` or `download_test` is set to `True` for the splits you want to use.

### No RTTM Files Found

```bash
WARNING: No RTTM files found for dev split
```

**Solution**: Check that the download completed successfully. Look for `.completed` marker in the dataset directory.

### Download Interrupted

```bash
Error downloading from https://www.robots.ox.ac.uk/~vgg/data/voxconverse/...
```

**Solution**:
 
- Check internet connection
- Set `force_download: true` to retry
- Manually download files and extract to correct directories

## Performance Considerations

- **Download Size**: 

  - Dev audio: ~1.5 GB
  - Test audio: ~1.7 GB
  - Annotations: ~1 MB
- **Processing Speed**: Fast (mainly I/O bound)
- **Memory Usage**: Low (processes files sequentially)

## Citation

If you use the VoxConverse dataset, please cite:

```bibtex
@article{chung2020spot,
  title={Spot the conversation: speaker diarisation in the wild},
  author={Chung, Joon Son and Huh, Jaesung and Nagrani, Arsha and Afouras, Triantafyllos and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2007.01216},
  year={2020}
}
```

## License

VoxConverse dataset is released under Creative Commons license. See the official website for complete license terms.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Custom Recipes](../../../recipes.md)
- [RTTM Format Specification](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v1.3.pdf)

