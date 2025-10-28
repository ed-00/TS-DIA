# AISHELL-4: Multi-channel Meeting Corpus

## Overview

AISHELL-4 is a sizable real-recorded Mandarin speech dataset collected by 8-channel circular microphone array for speech processing in conference scenarios. The dataset contains 211 recorded meeting sessions with a total of 120 hours of speech data, making it one of the largest publicly available Mandarin meeting corpora.

## Features

- **Mandarin Chinese**: Large-scale Mandarin meeting recordings
- **Multi-channel**: 8-channel circular microphone array
- **Real Meetings**: Authentic conference scenarios
- **Rich Annotations**: Speaker diarization and transcriptions
- **Lhotse Integration**: Uses Lhotse's built-in recipe

## Dataset Information

- **Name**: `aishell4`
- **Type**: Multi-channel meeting diarization and ASR
- **License**: Apache 2.0
- **Homepage**: https://www.aishelltech.com/aishell_4
- **Paper**: https://arxiv.org/abs/2104.03603
- **Duration**: ~120 hours
- **Language**: Mandarin Chinese

## Prerequisites

### System Dependencies
- **wget** or **curl**: For downloading audio files

### Python Dependencies
- `lhotse` (includes AISHELL-4 recipe)

## Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str/Path | **Required** | Directory to download dataset to |
| `force_download` | bool | `False` | Force re-download even if already present |
| `base_url` | str | `"http://www.openslr.org/resources"` | Base URL for downloads |

## Process Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str/Path | **Required** | Path to downloaded AISHELL-4 corpus |
| `output_dir` | str/Path | **Required** | Path to output manifests directory |
| `normalize_text` | bool | `False` | Normalize Chinese text |

## Configuration Example

### Minimal Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: aishell4
```

### Full Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: aishell4
    download_params:
      base_url: "http://www.openslr.org/resources"
    process_params:
      normalize_text: false
```

### With Text Normalization

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: aishell4
    process_params:
      normalize_text: true  # Normalize Chinese text
```

## Output Structure

```
./data/aishell4/                     # Downloaded dataset
├── train/                           # Training set
│   ├── wav/                         # Multi-channel audio
│   └── TextGrid/                    # Annotations
├── test/                            # Test set
└── .completed                       # Download completion marker

./manifests/aishell4/                # Generated manifests
├── aishell4_recordings_train.jsonl.gz
├── aishell4_supervisions_train.jsonl.gz
├── aishell4_recordings_test.jsonl.gz
└── aishell4_supervisions_test.jsonl.gz
```

## Usage Example

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/aishell4_config.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use for Mandarin diarization
for cut_set in cut_sets:
    print(f"Loaded {len(cut_set)} cuts")
    cut_set.describe()
```

### Command Line

```bash
# Process AISHELL-4 corpus
python -m data_manager.data_manager --config configs/aishell4_config.yml
```

## Dataset Statistics

- **Total Duration**: ~120 hours
- **Meetings**: 211 sessions
- **Speakers**: Multiple speakers per meeting
- **Channels**: 8-channel circular array
- **Sample Rate**: 16 kHz
- **Language**: Mandarin Chinese

## Splits

- **Train**: ~100 hours
- **Test**: ~20 hours

## Troubleshooting

### Download Fails
```
ERROR: Failed to download from openslr.org
```
**Solution**: Check internet connection. OpenSLR may have temporary outages.

### Chinese Text Issues
```
WARNING: Text encoding problems
```
**Solution**: Ensure Python environment supports UTF-8. Set `normalize_text: true`.

## Performance Considerations

- **Download Size**: ~15 GB
- **Processing Speed**: Moderate
- **Memory Usage**: Low to moderate
- **Multi-channel**: 8 channels per recording

## Use Cases

### Mandarin Meeting Diarization
```yaml
# Standard configuration for Mandarin diarization
normalize_text: false
```

### Mandarin ASR + Diarization
```yaml
# With text normalization for ASR
normalize_text: true
```

## Citation

```bibtex
@article{fu2021aishell,
  title={AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario},
  author={Fu, Yihui and Cheng, Luyao and Lv, Shubo and Jv, Yukai and Kong, Yuxuan and Chen, Zhuo and Hu, Yanxin and Xie, Lei and Wu, Jian and Bu, Hui and others},
  journal={arXiv preprint arXiv:2104.03603},
  year={2021}
}
```

## License

AISHELL-4 is released under Apache 2.0 license.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Lhotse AISHELL-4 Recipe](https://lhotse.readthedocs.io/en/latest/datasets.html#aishell-4)

