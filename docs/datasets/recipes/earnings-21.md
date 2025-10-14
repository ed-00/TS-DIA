# Earnings-21: Financial Earnings Call Dataset

## Overview

Earnings-21 is a 39-hour corpus of earnings calls containing entity-dense and numbers-heavy speech from nine different financial sectors. The dataset is designed for evaluating ASR and diarization systems on challenging financial domain speech with multiple speakers, technical terminology, and numerical data.

## Features

- **Financial Domain**: Earnings calls from various sectors
- **Entity-Dense Speech**: Rich in named entities and numbers
- **Multi-speaker**: Multiple speakers per call (analysts, executives)
- **Real-world Audio**: Authentic conference call recordings
- **Lhotse Integration**: Uses Lhotse's built-in recipe

## Dataset Information

- **Name**: `earnings21`
- **Type**: Financial domain diarization and ASR
- **License**: CC BY-SA 4.0
- **Homepage**: https://github.com/revdotcom/speech-datasets
- **Paper**: https://arxiv.org/abs/2104.11348
- **Duration**: ~39 hours
- **Language**: English

## Prerequisites

### System Dependencies
- **wget** or **curl**: For downloading audio files

### Python Dependencies
- `lhotse` (includes Earnings-21 recipe)

## Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str/Path | **Required** | Directory to download dataset to |
| `force_download` | bool | `False` | Force re-download even if already present |
| `url` | str | Auto-detected | Download URL (auto-configured) |

## Process Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str/Path | **Required** | Path to downloaded Earnings-21 corpus |
| `output_dir` | str/Path | **Required** | Path to output manifests directory |
| `normalize_text` | bool | `False` | Normalize text (lowercase, remove punctuation) |

## Configuration Example

### Minimal Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: earnings21
```

### Full Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: earnings21
    download_params:
      url: null  # Auto-detected
    process_params:
      normalize_text: false
```

### With Text Normalization

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: earnings21
    process_params:
      normalize_text: true  # Normalize for ASR
```

## Output Structure

```
./data/earnings21/                   # Downloaded dataset
├── media/                           # Audio files
│   ├── call001.mp3
│   ├── call002.mp3
│   └── ...
├── transcripts/                     # Transcriptions
│   ├── call001.json
│   ├── call002.json
│   └── ...
└── .completed                       # Download completion marker

./manifests/earnings21/              # Generated manifests
├── earnings21_recordings_train.jsonl.gz
├── earnings21_supervisions_train.jsonl.gz
├── earnings21_recordings_test.jsonl.gz
└── earnings21_supervisions_test.jsonl.gz
```

## Usage Example

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/earnings21_config.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use for financial domain diarization
for cut_set in cut_sets:
    print(f"Loaded {len(cut_set)} cuts")
    cut_set.describe()
```

### Command Line

```bash
# Process Earnings-21 corpus
python -m data_manager.data_manager --config configs/earnings21_config.yml
```

## Dataset Statistics

- **Total Duration**: ~39 hours
- **Earnings Calls**: Multiple calls from 9 sectors
- **Speakers**: Multiple per call (typically 2-10)
- **Sectors**: Technology, Healthcare, Finance, Retail, Energy, etc.
- **Sample Rate**: 16 kHz
- **Language**: English

## Financial Sectors

The dataset covers nine financial sectors:
- Technology
- Healthcare  
- Financial Services
- Consumer Goods
- Energy
- Industrials
- Materials
- Real Estate
- Utilities

## Troubleshooting

### Download Slow
```
Downloading earnings calls...
```
**Solution**: Dataset is ~5 GB. May take time depending on connection.

### Audio Format Issues
```
ERROR: Cannot decode MP3 files
```
**Solution**: Ensure ffmpeg is installed for MP3 decoding.

### Missing Transcripts
```
ERROR: Transcript files not found
```
**Solution**: Ensure complete download. Check for `.completed` marker.

## Performance Considerations

- **Download Size**: ~5 GB
- **Processing Speed**: Fast
- **Memory Usage**: Low
- **Audio Format**: MP3 (requires ffmpeg)

## Use Cases

### Financial Domain ASR
```yaml
# With text normalization for ASR
normalize_text: true
```

### Multi-speaker Diarization
```yaml
# Standard diarization setup
normalize_text: false
```

### Domain Adaptation
```yaml
# Use for financial domain adaptation
# Combine with general domain data
```

## Special Characteristics

### Entity-Dense Speech
- Company names
- Executive names
- Financial terms
- Product names

### Numbers-Heavy Content
- Revenue figures
- Percentages
- Dates and quarters
- Financial metrics

### Multiple Speaker Types
- Company executives (CEO, CFO)
- Financial analysts
- Moderators
- Questioners

## Citation

```bibtex
@article{delrio2021earnings,
  title={Earnings-21: A Practical Benchmark for ASR in the Wild},
  author={Del Rio, Miguel and Delworth, Natalie and Westerman, Ryan and Huang, Michelle and Bhandari, Nishchal and Palakapilly, Joseph and McNamara, Quinten and Dong, Joshua and Zelasko, Piotr and Jett, Miguel},
  journal={arXiv preprint arXiv:2104.11348},
  year={2021}
}
```

## License

Earnings-21 is released under CC BY-SA 4.0 license.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Lhotse Earnings-21 Recipe](https://lhotse.readthedocs.io/en/latest/datasets.html#earnings-21)

