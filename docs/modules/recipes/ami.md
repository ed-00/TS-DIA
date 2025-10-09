# AMI Meeting Corpus

## Overview

The AMI (Augmented Multi-party Interaction) Meeting Corpus is a multi-modal dataset of meeting recordings. It contains 100 hours of meeting recordings with high-quality transcriptions and speaker diarization annotations. The corpus includes both scenario-based and natural meetings recorded in multiple locations.

## Features

- **Meeting Recordings**: Real and scenario-based meetings
- **Multiple Microphones**: Individual headset (IHM), single distant (SDM), and multiple distant (MDM)
- **Rich Annotations**: Transcriptions, speaker diarization, dialogue acts
- **Lhotse Integration**: Uses Lhotse's built-in recipe
- **Multiple Partitions**: Full corpus, scenario-only, or custom splits

## Dataset Information

- **Name**: `ami`
- **Type**: Meeting diarization and transcription
- **License**: CC BY 4.0
- **Homepage**: https://groups.inf.ed.ac.uk/ami/corpus/
- **Paper**: https://link.springer.com/article/10.1007/s10579-012-9190-y
- **Duration**: ~100 hours
- **Speakers**: Multiple speakers per meeting (typically 4)

## Prerequisites

### System Dependencies
- **wget** or **curl**: For downloading audio files
- **ffmpeg**: May be needed for audio processing

### Python Dependencies
- `lhotse` (includes AMI recipe)

## Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str/Path | **Required** | Directory to download dataset to |
| `force_download` | bool | `False` | Force re-download even if already present |
| `annotations` | str/Path | `None` | Path to annotations (auto-downloaded if None) |
| `url` | str | `"http://groups.inf.ed.ac.uk/ami"` | Base URL for downloads |
| `mic` | str | `"ihm"` | Microphone type: "ihm", "sdm", or "mdm" |

### Microphone Types

- **ihm** (Individual Headset Microphone): Best quality, one mic per speaker
- **sdm** (Single Distant Microphone): Single room microphone
- **mdm** (Multiple Distant Microphone): Multiple room microphones (array)

## Process Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str/Path | **Required** | Path to downloaded AMI corpus |
| `output_dir` | str/Path | **Required** | Path to output manifests directory |
| `data_dir` | str/Path | `None` | Alternative to corpus_dir |
| `annotations_dir` | str/Path | `None` | Path to annotations directory |
| `mic` | str | `"ihm"` | Microphone type to process |
| `partition` | str | `"full-corpus"` | Partition: "full-corpus", "full-corpus-asr", "scenario-only" |
| `normalize_text` | str | `"kaldi"` | Text normalization: "none", "upper", "kaldi" |
| `max_words_per_segment` | int | `None` | Split long segments (None = no splitting) |
| `merge_consecutive` | bool | `False` | Merge consecutive same-speaker segments |

### Available Partitions

- **full-corpus**: All meetings (100 hours)
- **full-corpus-asr**: Optimized for ASR tasks
- **scenario-only**: Only scenario-based meetings (~70 hours)

## Configuration Example

### Minimal Configuration (IHM)

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: ami
```

### Full Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: ami
    download_params:
      mic: "ihm"
      url: "http://groups.inf.ed.ac.uk/ami"
      annotations: null
    process_params:
      mic: "ihm"
      partition: "full-corpus"
      normalize_text: "kaldi"
      max_words_per_segment: null
      merge_consecutive: false
```

### Distant Microphone Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: ami
    download_params:
      mic: "mdm"  # Multiple distant microphones
    process_params:
      mic: "mdm"
      partition: "full-corpus"
      normalize_text: "kaldi"
```

### Scenario-Only with Merging

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: ami
    process_params:
      mic: "ihm"
      partition: "scenario-only"
      normalize_text: "kaldi"
      merge_consecutive: true  # Merge same-speaker segments
```

## Output Structure

```
./data/ami/                          # Downloaded dataset
├── annotations/                     # Annotations
│   ├── words/
│   ├── segments/
│   └── ...
├── audio/                           # Audio files
│   ├── ihm/                         # Individual headset mics
│   ├── sdm/                         # Single distant mic
│   └── mdm/                         # Multiple distant mics
└── .completed                       # Download completion marker

./manifests/ami/                     # Generated manifests
├── ami_recordings_train.jsonl.gz
├── ami_supervisions_train.jsonl.gz
├── ami_recordings_dev.jsonl.gz
├── ami_supervisions_dev.jsonl.gz
├── ami_recordings_test.jsonl.gz
└── ami_supervisions_test.jsonl.gz
```

## Usage Example

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/ami_config.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use for diarization
for cut_set in cut_sets:
    print(f"Loaded {len(cut_set)} cuts")
    cut_set.describe()
```

### Command Line

```bash
# Process AMI corpus
python -m data_manager.data_manager --config configs/ami_config.yml
```

## Dataset Statistics

- **Total Duration**: ~100 hours
- **Meetings**: 171 meetings
- **Speakers**: ~150 unique speakers
- **Average Meeting Length**: ~35 minutes
- **Speakers per Meeting**: Typically 4
- **Languages**: English (with various accents)

## Splits

- **Train**: ~80 hours
- **Dev**: ~10 hours
- **Test**: ~10 hours

## Troubleshooting

### Download Slow
```
Downloading from groups.inf.ed.ac.uk...
```
**Solution**: AMI corpus is large. Download may take several hours. Consider downloading overnight.

### Missing Annotations
```
ERROR: Annotations not found
```
**Solution**: Lhotse automatically downloads annotations. Ensure internet connection is stable.

### Microphone Type Mismatch
```
ERROR: No audio files found for mic type
```
**Solution**: Ensure `mic` parameter matches in both download_params and process_params.

## Performance Considerations

- **Download Size**:
  - IHM: ~15 GB
  - SDM: ~5 GB
  - MDM: ~20 GB
- **Processing Speed**: Moderate (depends on partition size)
- **Memory Usage**: Low to moderate

## Use Cases

### Speaker Diarization
```yaml
# Use IHM for clean diarization
mic: "ihm"
partition: "full-corpus"
merge_consecutive: true
```

### Far-field Diarization
```yaml
# Use MDM for realistic far-field scenarios
mic: "mdm"
partition: "full-corpus"
```

### ASR + Diarization
```yaml
# Use ASR-optimized partition
partition: "full-corpus-asr"
normalize_text: "kaldi"
```

## Citation

```bibtex
@article{carletta2007ami,
  title={The AMI meeting corpus: A pre-announcement},
  author={Carletta, Jean and Ashby, Simone and Bourban, Sebastien and Flynn, Mike and Guillemot, Mael and Hain, Thomas and Kadlec, Jaroslav and Karaiskos, Vasilis and Kraaij, Wessel and Kronenthal, Melissa and others},
  journal={Machine learning for multimodal interaction},
  pages={28--39},
  year={2007},
  publisher={Springer}
}
```

## License

AMI Meeting Corpus is released under CC BY 4.0 license.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Lhotse AMI Recipe](https://lhotse.readthedocs.io/en/latest/datasets.html#ami)

