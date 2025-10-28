# MSDWILD: Multi-modal Speaker Diarization Dataset in the Wild

## Overview

MSDWILD (Multi-modal Speaker Diarization Dataset in the Wild) is a multi-modal speaker diarization dataset designed for speaker diarization and lip-speech synchronization in the wild. The dataset includes audio files with RTTM annotations, and optionally video files and cropped faces for multi-modal research.

## Features

- **Multi-modal Data**: Audio, video, and face crops
- **RTTM Annotations**: Standard RTTM format for diarization
- **Multiple Splits**: Train, dev, and test splits
- **Google Drive Download**: Audio files hosted on Google Drive
- **Automatic Retry**: Robust download with retry logic

## Dataset Information

- **Name**: `mswild`
- **Type**: Multi-modal speaker diarization
- **License**: Check dataset repository for license terms
- **Homepage**: https://github.com/X-LANCE/MSDWILD
- **Paper**: TBD (check repository for citation)
- **Splits**: few_train, few_val, many_val

## Prerequisites

### System Dependencies

- None (pure Python download and processing)

### Python Dependencies

- `lhotse` (for manifest generation)
- `gdown` (for Google Drive downloads)
- `tqdm` (for progress bars)

## Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str/Path | **Required** | Directory to download dataset to |
| `force_download` | bool | `False` | Force re-download even if already present |
| `download_audio` | bool | `True` | Download audio files (~7.56 GB, required) |
| `download_video` | bool | `False` | Download video files (~43.14 GB, manual) |
| `download_faces` | bool | `False` | Download cropped faces (~14.49 GB, manual) |

**Note**: Video and face downloads require manual download from Google Drive links provided in the repository.

## Process Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str/Path | **Required** | Path to downloaded MSDWILD dataset |
| `output_dir` | str/Path | `None` | Path to output manifests directory |
| `splits` | Dict[str, str] | `{"train": "few_train", "dev": "few_val", "test": "many_val"}` | Split name to RTTM pattern mapping |

### Available Split Patterns

- **few_train**: Few-speaker training set
- **few_val**: Few-speaker validation set
- **many_val**: Many-speaker validation set

## Configuration Example

### Minimal Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: mswild
```

### Full Configuration (Audio Only)

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: mswild
    download_params:
      force_download: false
      download_audio: true
      download_video: false
      download_faces: false
    process_params:
      splits:
        train: "few_train"
        dev: "few_val"
        test: "many_val"
```

### Custom Splits Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: mswild
    download_params:
      download_audio: true
    process_params:
      splits:
        train: "few_train"
        val: "few_val"
```

## Output Structure

```
./data/mswild/                       # Downloaded dataset
├── wavs/                            # Audio files
│   └── wav/
│       ├── file001.wav
│       ├── file002.wav
│       └── ...
├── rttms/                           # RTTM annotations
│   ├── few_train.rttm
│   ├── few_val.rttm
│   ├── many_val.rttm
│   └── ...
└── .completed                       # Download completion marker

./manifests/mswild/                  # Generated manifests
├── msdwild_recordings_train.jsonl.gz
├── msdwild_supervisions_train.jsonl.gz
├── msdwild_recordings_dev.jsonl.gz
├── msdwild_supervisions_dev.jsonl.gz
├── msdwild_recordings_test.jsonl.gz
└── msdwild_supervisions_test.jsonl.gz
```

## Usage Example

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/msdwild_config.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use for diarization
for cut_set in cut_sets:
    print(f"Loaded {len(cut_set)} cuts")
    cut_set.describe()
```

### Direct Recipe Usage

```python
from data_manager.recipes.mswild import download_mswild, prepare_mswild

# Download (audio only)
corpus_dir = download_mswild(
    target_dir="./data",
    download_audio=True,
    download_video=False,
    download_faces=False
)

# Process
manifests = prepare_mswild(
    corpus_dir=corpus_dir,
    output_dir="./manifests/mswild"
)

# Access manifests
train_recordings = manifests["train"]["recordings"]
train_supervisions = manifests["train"]["supervisions"]
```

### Command Line

```bash
# Process MSDWILD dataset
python -m data_manager.data_manager --config configs/msdwild_config.yml
```

## RTTM Format

MSDWILD uses standard RTTM (Rich Transcription Time Marked) format:

```
SPEAKER file_id 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
```

Example:
```
SPEAKER file001 1 0.50 1.20 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER file001 1 1.80 2.50 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER file002 1 0.00 3.50 <NA> <NA> SPEAKER_00 <NA> <NA>
```

## Download Details

### Audio Files (Google Drive)
- **Size**: ~7.56 GB
- **Format**: WAV files
- **Download Method**: Automatic using `gdown`
- **Retry Logic**: 3 attempts with automatic retry

### Manual Downloads
For video and face data:
1. Visit the MSDWILD GitHub repository
2. Follow Google Drive links for video/faces
3. Extract to appropriate subdirectories in the dataset folder

## Troubleshooting

### Google Drive Download Quota Exceeded
```
gdown.exceptions.FileURLRetrievalError: Failed to retrieve file url
```
**Solution**:
- Wait 24 hours for quota reset
- Manually download from: https://drive.google.com/file/d/1I5qfuPPGBM9keJKz0VN-OYEeRMJ7dgpl
- Extract to `./data/mswild/wavs/`

### Missing Audio Files
```
WARNING: Audio file not found: .../mswild/wavs/wav/xxx.wav
```
**Solution**: Ensure audio download completed. Check for `.completed` marker and verify wavs/wav/ directory exists.

### Download Interrupted
```
Download attempt 3 failed
```
**Solution**:
- Set `force_download: true` to retry
- Check internet connection
- Manually download and extract files

### No RTTM Files Found
```
WARNING: No RTTM files found for train split
```
**Solution**: Verify RTTM files are in `rttms/` directory with expected patterns (few_train, few_val, many_val).

## Performance Considerations

- **Download Size**: 
  - Audio only: ~7.56 GB
  - With video: ~50 GB total
  - With all modalities: ~65 GB total
- **Processing Speed**: Fast (mainly I/O bound)
- **Memory Usage**: Low (sequential processing)
- **Google Drive**: May encounter quota limits for large downloads

## Dataset Statistics

### Few-speaker Splits
- **few_train**: Training set with fewer speakers per recording
- **few_val**: Validation set with fewer speakers per recording

### Many-speaker Split
- **many_val**: Validation set with many speakers per recording (more challenging)

## Multi-modal Extensions

While this recipe focuses on audio diarization, the dataset also provides:
- **Video Files**: Original video recordings for audio-visual diarization
- **Face Crops**: Pre-extracted face crops for each speaker segment
- **Lip Synchronization**: Annotations for lip-speech sync research

## Citation

```bibtex
# Check the MSDWILD repository for the official citation
# https://github.com/X-LANCE/MSDWILD
```

## License

MSDWILD dataset license terms are available in the repository. Please review before use.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Custom Recipes](../../../recipes.md)
- [RTTM Format Specification](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v1.3.pdf)

