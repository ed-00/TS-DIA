# ICSI Meeting Corpus

## Overview

The ICSI (International Computer Science Institute) Meeting Corpus consists of naturally occurring meetings recorded at ICSI in Berkeley. The corpus contains 75 meetings with high-quality close-talking microphone recordings and detailed transcriptions, making it ideal for meeting diarization and transcription research.

## Features

- **Natural Meetings**: Real research group meetings
- **Multiple Microphones**: Individual headset (IHM) and distant array microphones
- **Rich Transcriptions**: Word-level timestamps and speaker labels
- **Lhotse Integration**: Uses Lhotse's built-in recipe
- **Long Meetings**: Average meeting duration ~60 minutes

## Dataset Information

- **Name**: `icsi`
- **Type**: Meeting diarization and transcription
- **License**: ICSI Meeting Corpus License
- **Homepage**: https://groups.inf.ed.ac.uk/ami/icsi/
- **Paper**: https://www.isca-speech.org/archive/icslp_2004/janin04_icslp.html
- **Duration**: ~72 hours
- **Speakers**: ~50 unique speakers

## Prerequisites

### System Dependencies
- **wget** or **curl**: For downloading audio files
- **sox**: For audio format conversion (optional)

### Python Dependencies
- `lhotse` (includes ICSI recipe)

## Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str/Path | **Required** | Directory to download dataset to |
| `force_download` | bool | `False` | Force re-download even if already present |
| `audio_dir` | str/Path | `None` | Path to audio directory (auto-downloaded if None) |
| `transcripts_dir` | str/Path | `None` | Path to transcripts (auto-downloaded if None) |
| `url` | str | `"http://groups.inf.ed.ac.uk/ami"` | Base URL for downloads |
| `mic` | str | `"ihm"` | Microphone type: "ihm" or "mdm" |

### Microphone Types

- **ihm** (Individual Headset Microphone): Close-talking, best quality
- **mdm** (Multiple Distant Microphone): Far-field array microphones

## Process Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str/Path | **Required** | Path to downloaded ICSI corpus |
| `output_dir` | str/Path | **Required** | Path to output manifests directory |
| `audio_dir` | str/Path | `None` | Path to audio directory |
| `transcripts_dir` | str/Path | `None` | Path to transcripts directory |
| `mic` | str | `"ihm"` | Microphone type to process |
| `normalize_text` | str | `"kaldi"` | Text normalization: "none", "upper", "kaldi" |
| `save_to_wav` | bool | `False` | Convert SPH files to WAV format |

## Configuration Example

### Minimal Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: icsi
```

### Full Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: icsi
    download_params:
      mic: "ihm"
      url: "http://groups.inf.ed.ac.uk/ami"
      audio_dir: null
      transcripts_dir: null
    process_params:
      mic: "ihm"
      normalize_text: "kaldi"
      save_to_wav: false
```

### WAV Conversion Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: icsi
    process_params:
      mic: "ihm"
      normalize_text: "kaldi"
      save_to_wav: true  # Convert SPH to WAV
```

## Output Structure

```
./data/icsi/                         # Downloaded dataset
├── audio/                           # Audio files
│   ├── ihm/                         # Individual headset mics
│   └── mdm/                         # Multiple distant mics
├── transcripts/                     # Transcription files
│   ├── Bed004.mrt
│   ├── Bed009.mrt
│   └── ...
└── .completed                       # Download completion marker

./manifests/icsi/                    # Generated manifests
├── icsi_recordings_train.jsonl.gz
├── icsi_supervisions_train.jsonl.gz
├── icsi_recordings_dev.jsonl.gz
├── icsi_supervisions_dev.jsonl.gz
├── icsi_recordings_test.jsonl.gz
└── icsi_supervisions_test.jsonl.gz
```

## Usage Example

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/icsi_config.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use for diarization
for cut_set in cut_sets:
    print(f"Loaded {len(cut_set)} cuts")
    cut_set.describe()
```

### Command Line

```bash
# Process ICSI corpus
python -m data_manager.data_manager --config configs/icsi_config.yml
```

## Dataset Statistics

- **Total Duration**: ~72 hours
- **Meetings**: 75 meetings
- **Speakers**: ~50 unique speakers
- **Average Meeting Length**: ~60 minutes
- **Speakers per Meeting**: 3-9 (average ~6)
- **Language**: English

## Meeting Types

- **Research group meetings**: Regular team meetings
- **Project discussions**: Focused technical discussions
- **Reading groups**: Paper discussion meetings
- **Brainstorming sessions**: Idea generation meetings

## Troubleshooting

### Download Slow
```
Downloading from groups.inf.ed.ac.uk...
```
**Solution**: ICSI corpus is large. Download may take several hours.

### SPH Format Issues
```
ERROR: Cannot read SPH audio files
```
**Solution**: Set `save_to_wav: true` to convert SPH files to WAV format.

### Missing Transcripts
```
ERROR: Transcript files not found
```
**Solution**: Ensure transcripts are downloaded. Lhotse handles this automatically.

## Performance Considerations

- **Download Size**:
  - IHM: ~10 GB
  - MDM: ~15 GB
- **Processing Speed**: Moderate
- **Memory Usage**: Low to moderate
- **SPH to WAV Conversion**: Requires additional disk space

## Use Cases

### Meeting Diarization
```yaml
# Use IHM for clean meeting diarization
mic: "ihm"
normalize_text: "kaldi"
```

### Far-field Meeting Analysis
```yaml
# Use MDM for realistic far-field scenarios
mic: "mdm"
save_to_wav: true
```

## Citation

```bibtex
@inproceedings{janin2003icsi,
  title={The ICSI meeting corpus},
  author={Janin, Adam and Baron, Don and Edwards, Jane and Ellis, Dan and Gelbart, David and Morgan, Nelson and Peskin, Barbara and Pfau, Thilo and Shriberg, Elizabeth and Stolcke, Andreas and others},
  booktitle={2003 IEEE International Conference on Acoustics, Speech, and Signal Processing},
  volume={1},
  pages={I--I},
  year={2003},
  organization={IEEE}
}
```

## License

ICSI Meeting Corpus has its own license. Please review before use.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Lhotse ICSI Recipe](https://lhotse.readthedocs.io/en/latest/datasets.html#icsi)

