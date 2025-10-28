# AVA-AVD: Active Speaker Detection Dataset

## Overview

AVA-AVD (AVA Active Video Detection) is a dataset for active speaker detection, containing video clips with annotations for who is speaking when. The dataset is derived from the AVA (Atomic Visual Actions) dataset and includes both visual and audio information for speaker diarization research.

## Features

- **Video + Audio**: Multi-modal data for active speaker detection
- **Multiple Splits**: Train, validation, and test sets
- **Lab/RTTM Annotations**: Support for both annotation formats
- **AWS Hosting**: Videos hosted on AWS S3
- **Google Drive Annotations**: Annotations available via Google Drive

## Dataset Information

- **Name**: `ava_avd`
- **Type**: Active speaker detection / Multi-modal diarization
- **License**: AVA Dataset License (check official website)
- **Homepage**: https://github.com/zcxu-eric/AVA-AVD
- **Paper**: https://arxiv.org/abs/1901.01342
- **Splits**: train, val, test

## Prerequisites

### System Dependencies

- **wget**: For downloading videos

  ```bash
  sudo apt-get install wget  # Ubuntu/Debian
  brew install wget          # macOS
  ```

### Python Dependencies
- `lhotse` (for manifest generation)
- `gdown` (for Google Drive downloads)
- `tqdm` (for progress bars)

## Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str/Path | **Required** | Directory to download dataset to |
| `force_download` | bool | `False` | Force re-download even if already present |
| `download_annotations` | bool | `True` | Download annotation files from Google Drive |
| `download_videos` | bool | `True` | Download video files from AWS S3 |

## Process Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str/Path | **Required** | Path to downloaded AVA-AVD dataset |
| `output_dir` | str/Path | `None` | Path to output manifests directory |
| `splits` | Dict[str, str] | `{"train": "train", "val": "val", "test": "test"}` | Split name to directory mapping |

## Configuration Example

### Minimal Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: ava_avd
```

### Full Configuration

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: ava_avd
    download_params:
      force_download: false
      download_annotations: true
      download_videos: true
    process_params:
      splits:
        train: "train"
        val: "val"
        test: "test"
```

### Annotations Only (No Videos)

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: ava_avd
    download_params:
      download_annotations: true
      download_videos: false
```

## Output Structure

```
./data/ava_avd/                      # Downloaded dataset
├── dataset/
│   ├── videos/                      # Video files
│   │   ├── video001.mp4
│   │   ├── video002.mp4
│   │   └── ...
│   ├── annotations/                 # Annotations
│   │   ├── train/
│   │   │   ├── file001.lab
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   └── split/
│       └── video.list               # List of videos to download
└── .completed                       # Download completion marker

./manifests/ava_avd/                 # Generated manifests
├── ava_avd_recordings_train.jsonl.gz
├── ava_avd_supervisions_train.jsonl.gz
├── ava_avd_recordings_val.jsonl.gz
├── ava_avd_supervisions_val.jsonl.gz
├── ava_avd_recordings_test.jsonl.gz
└── ava_avd_supervisions_test.jsonl.gz
```

## Usage Example

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/ava_avd_config.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use for active speaker detection
for cut_set in cut_sets:
    print(f"Loaded {len(cut_set)} cuts")
    cut_set.describe()
```

### Command Line

```bash
# Process AVA-AVD dataset
python -m data_manager.data_manager --config configs/ava_avd_config.yml
```

## Annotation Formats

### Lab Format (.lab)
Simple format with start time, end time, and speaker label:
```
0.5 1.2 SPEAKER_00
1.8 2.5 SPEAKER_01
4.5 5.0 SPEAKER_00
```

### RTTM Format (.rttm)
Standard RTTM format also supported:
```
SPEAKER file_id 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
```

## Troubleshooting

### wget Not Found
```
sh: wget: command not found
```
**Solution**: Install wget using your package manager.

### Google Drive Download Fails
```
gdown: Failed to download annotations
```
**Solution**: 
- Check internet connection
- Verify Google Drive ID is correct
- Try manual download from: https://drive.google.com/file/d/18kjJJbebBg7e8umI6HoGE4_tI3OWufzA

### Video Download Slow
```
Videos downloading very slowly...
```
**Solution**: 
- AWS S3 download speeds depend on your location
- Consider downloading overnight for large datasets
- Use `download_videos: false` and download videos separately if needed

### Missing Video List
```
WARNING: Video list file not found
```
**Solution**: Ensure repository downloaded correctly. Check for `dataset/split/video.list` file.

## Performance Considerations

- **Download Size**: 
  - Annotations: ~100 MB
  - Videos: Variable (depends on video.list)
  - Total: Several GB
- **Processing Speed**: Fast for annotations, slower for videos
- **Network**: Requires stable connection for video downloads
- **Storage**: Plan for significant storage if downloading all videos

## Citation

```bibtex
@inproceedings{roth2020ava,
  title={Ava active speaker: An audio-visual dataset for active speaker detection},
  author={Roth, Joseph and Chaudhuri, Sourish and Klejch, Ondrej and Marvin, Radhika and Gallagher, Andrew and Kaver, Lisa and Ramaswamy, Sharadh and Stopczynski, Arkadiusz and Schmid, Cordelia and Xi, Zhonghua and others},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4492--4496},
  year={2020},
  organization={IEEE}
}
```

## License

AVA-AVD dataset follows the AVA dataset license. Please review the official license before use.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Custom Recipes](../../../recipes.md)

