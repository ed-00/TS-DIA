# Ego4D Audio Diarization Recipe

## Overview

The Ego4D dataset is a large-scale egocentric (first-person) video dataset that includes audio-visual data from first-person perspectives. This recipe focuses on extracting and processing audio data for speaker diarization tasks.

## Features

- **Audio Extraction**: Automatic extraction of audio from video files using ffmpeg
- **Voice Activity Processing**: Processing of voice activity annotations from Ego4D JSON/JSONL files
- **AWS Integration**: Secure AWS credential management for dataset access
- **Annotation Subset Filtering**: Support for targeting specific annotation types (e.g., "av" for audio-visual)
- **Auto CLI Installation**: Automatically installs Ego4D CLI if not available
- **Flexible Dataset Parts**: Download only the parts you need (clips, annotations, etc.)

## Dataset Information

- **Name**: `ego4d`
- **Type**: Egocentric video with audio diarization annotations
- **License**: Ego4D License (requires registration and AWS credentials)
- **Homepage**: https://ego4d-data.org/
- **Paper**: https://arxiv.org/abs/2110.07058

## Prerequisites

### 1. Ego4D Registration
Register at https://ego4d-data.org/ to obtain AWS credentials for dataset access.

### 2. AWS Credentials
Set your AWS credentials (obtained from Ego4D registration):

```bash
export AWS_ACCESS_KEY_ID="your_aws_access_key_id"
export AWS_ACCESS_KEY="your_aws_secret_access_key"
```

Or create a `.env` file:
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_ACCESS_KEY=your_aws_secret_access_key
```

### 3. System Dependencies
- **ffmpeg**: Required for audio extraction from video files
  ```bash
  sudo apt-get install ffmpeg  # Ubuntu/Debian
  brew install ffmpeg          # macOS
  ```

### 4. Python Dependencies
- `ego4d` CLI (auto-installed if `install_cli: true`)
- `python-dotenv` (for .env file support)
- `lhotse` (for manifest generation)

## Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str/Path | **Required** | Directory to download dataset to |
| `access_key` | str | `None` | AWS access key (if not in environment) |
| `force_download` | bool | `False` | Force re-download even if already present |
| `dataset_parts` | List[str] | `["clips", "annotations"]` | Parts to download |
| `install_cli` | bool | `True` | Auto-install Ego4D CLI if not available |
| `timeout` | int | `3600` | Download timeout in seconds |
| `env_file` | str/Path | `None` | Path to .env file for credentials |

### Available Dataset Parts

- `clips`: Video clips (required for audio extraction)
- `annotations`: Diarization annotations (voice activity)
- `metadata`: Dataset metadata
- `takes`: Full-length video takes
- `captures`: Camera captures
- And more (see `Ego4DPart` enum in source)

## Process Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str/Path | **Required** | Path to downloaded Ego4D dataset |
| `output_dir` | str/Path | **Required** | Path to output manifests directory |
| `extract_audio` | bool | `True` | Extract audio from video files |
| `audio_sample_rate` | int | `16000` | Sample rate for extracted audio (Hz) |
| `min_segment_duration` | float | `0.5` | Minimum segment duration (seconds) |
| `max_segment_duration` | float | `30.0` | Maximum segment duration (seconds) |
| `max_clips` | int | `None` | Limit number of clips to process (0 = no limit) |
| `annotation_subset` | str | `None` | Filter annotations by type (e.g., "av" for audio-visual) |

## Configuration Example

### Minimal Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: ego4d
```

### Full Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: ego4d
    download_params:
      # AWS credentials loaded from environment or .env
      dataset_parts: ["clips", "annotations"]
      install_cli: true
      timeout: 3600  # 1 hour
      env_file: "./.env"
    process_params:
      extract_audio: true
      audio_sample_rate: 16000
      min_segment_duration: 0.5
      max_segment_duration: 30.0
      max_clips: 0  # Process all clips
      annotation_subset: "av"  # Audio-visual annotations only
```

### Small Test Configuration

```yaml
# Test with limited clips for quick validation
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: true

datasets:
  - name: ego4d
    download_params:
      dataset_parts: ["clips", "annotations"]
      install_cli: true
      timeout: 1800  # 30 minutes
      env_file: "./.env"
    process_params:
      extract_audio: true
      audio_sample_rate: 16000
      min_segment_duration: 0.5
      max_segment_duration: 30.0
      max_clips: 10  # Only process 10 clips for testing
      annotation_subset: "av"
```

## Output Structure

```
./data/ego4d/                        # Downloaded dataset
├── v2/                              # Ego4D v2 structure
│   ├── clips/                       # Video clips
│   │   ├── 00a21a4a-50cb-46d8-a8c0-bc637a4d747e.mp4
│   │   └── ...
│   └── annotations/                 # Annotations
│       ├── av_train.json
│       ├── av_val.json
│       └── ...
└── .ego4d_completed                 # Download completion marker

./manifests/ego4d/                   # Generated manifests
├── 00a21a4a-50cb-46d8-a8c0-bc637a4d747e.wav  # Extracted audio
├── ego4d_recordings.jsonl.gz        # Recording manifest
├── ego4d_supervisions.jsonl.gz      # Supervision manifest
└── ego4d_cuts.jsonl.gz             # Cut manifest
```

## Usage Example

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/ego4d_config.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use for diarization
for cut_set in cut_sets:
    print(f"Loaded {len(cut_set)} cuts")
    cut_set.describe()
```

### Command Line

```bash
# Process Ego4D dataset
python -m data_manager.data_manager --config configs/ego4d_config.yml

# Set credentials first
export AWS_ACCESS_KEY_ID="your_key_id"
export AWS_ACCESS_KEY="your_secret_key"
```

## Annotation Format

The Ego4D annotations include voice activity segments with:

- **start_time**: Segment start time in seconds
- **end_time**: Segment end time in seconds
- **speaker_id**: Speaker identifier
- **confidence**: Annotation confidence (0.0-1.0)
- **video_uid**: Video clip identifier

Example annotation structure:
```json
{
  "video_uid": "00a21a4a-50cb-46d8-a8c0-bc637a4d747e",
  "voice_activity": [
    {
      "start_time": 1.5,
      "end_time": 3.2,
      "speaker_id": "speaker_01",
      "confidence": 0.95
    }
  ]
}
```

## Audio Extraction

The recipe automatically extracts audio from video files using ffmpeg with:
- **Codec**: PCM 16-bit signed little-endian
- **Sample Rate**: 16000 Hz (configurable)
- **Channels**: 1 (mono)
- **Format**: WAV

## Troubleshooting

### AWS Credentials Error
```
ValueError: AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_ACCESS_KEY
```
**Solution**: Set environment variables or create `.env` file with credentials.

### 403 Forbidden Error
```
botocore.exceptions.ClientError: An error occurred (403) when calling the HeadObject operation: Forbidden
```
**Solution**: 
- Verify your AWS credentials are valid
- Ensure you've registered at https://ego4d-data.org/
- Check credentials haven't expired

### ffmpeg Not Found
```
FileNotFoundError: ffmpeg not found
```
**Solution**: Install ffmpeg using your package manager.

### Ego4D CLI Not Installed
```
RuntimeError: Ego4D CLI not available and installation disabled
```
**Solution**: Set `install_cli: true` or manually install: `pip install ego4d`

## Performance Considerations

- **Download Size**: Full Ego4D clips dataset is several TB. Use `max_clips` for testing.
- **Audio Extraction**: CPU-intensive. Expect ~1-2x real-time extraction speed.
- **Storage**: Extracted audio requires additional space (~10% of video size).
- **Network**: Use stable, high-bandwidth connection for large downloads.

## Citation

If you use the Ego4D dataset, please cite:

```bibtex
@inproceedings{grauman2022ego4d,
  title={Ego4d: Around the world in 3,000 hours of egocentric video},
  author={Grauman, Kristen and Westbury, Andrew and Byrne, Eugene and Chavis, Zachary and Furnari, Antonino and Girdhar, Rohit and Hamburger, Jackson and Jiang, Hao and Liu, Miao and Liu, Xingyu and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18995--19012},
  year={2022}
}
```

## License

Ego4D dataset is subject to its own license terms. See https://ego4d-data.org/license/ for details.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Custom Recipes](../../../recipes.md)

