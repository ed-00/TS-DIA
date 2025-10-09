# Dataset Recipes Documentation

This directory contains comprehensive documentation for all dataset recipes supported by the TS-DIA data manager.

## Overview

The data manager supports two types of datasets:

1. **Custom Recipes**: Datasets with custom download and processing implementations
2. **Lhotse-Based Recipes**: Datasets using Lhotse's built-in recipes

## Custom Recipes

These datasets have custom implementations in `data_manager/recipes/`:

### [Ego4D Audio Diarization](ego4d.md)
- **Type**: Egocentric video with audio diarization
- **Size**: Variable (clips-based)
- **Features**: Audio extraction from video, voice activity annotations
- **Use Case**: First-person audio diarization
- **Status**: ✅ Implemented

### [VoxConverse](voxconverse.md)
- **Type**: Speaker diarization from YouTube
- **Size**: ~32 hours (dev + test)
- **Features**: RTTM annotations, multi-speaker conversations
- **Use Case**: Speaker diarization evaluation
- **Status**: ✅ Implemented

### [MSDWild](mswild.md)
- **Type**: Multi-modal speaker diarization
- **Size**: ~7.56 GB audio + optional video/faces
- **Features**: RTTM annotations, multi-modal data
- **Use Case**: Multi-modal diarization research
- **Status**: ✅ Implemented

### [AVA-AVD](ava-avd.md)
- **Type**: Active speaker detection
- **Size**: Variable (video-based)
- **Features**: Audio-visual active speaker annotations
- **Use Case**: Active speaker detection, multi-modal diarization
- **Status**: ✅ Implemented

### [LibriheavyMix](libriheavy-mix.md)
- **Type**: Overlapped speech synthesis
- **Size**: 100h (small) to 20,000h (full)
- **Features**: 1-4 speaker mixtures, reverberation, SNR augmentation
- **Use Case**: Overlapped speech diarization training
- **Status**: ✅ Implemented

## Lhotse-Based Recipes

These datasets use Lhotse's built-in recipes with parameter configurations:

### [AMI Meeting Corpus](ami.md)
- **Type**: Meeting recordings
- **Size**: ~100 hours
- **Features**: Multiple microphone types (IHM/SDM/MDM), rich annotations
- **Use Case**: Meeting diarization and transcription
- **Status**: ✅ Supported via Lhotse

### [ICSI Meeting Corpus](icsi.md)
- **Type**: Natural meeting recordings
- **Size**: ~72 hours
- **Features**: Close-talking and distant microphones
- **Use Case**: Meeting diarization research
- **Status**: ✅ Supported via Lhotse

### [AISHELL-4](aishell-4.md)
- **Type**: Mandarin meeting corpus
- **Size**: ~120 hours
- **Features**: 8-channel circular array, Mandarin Chinese
- **Use Case**: Mandarin meeting diarization
- **Status**: ✅ Supported via Lhotse

### [Earnings-21](earnings-21.md)
- **Type**: Financial earnings calls
- **Size**: ~39 hours
- **Features**: Entity-dense, numbers-heavy speech
- **Use Case**: Financial domain diarization and ASR
- **Status**: ✅ Supported via Lhotse

## Quick Start

### Using Configuration Files

All datasets can be configured via YAML:

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: voxconverse
  - name: ami
  - name: ego4d
    download_params:
      dataset_parts: ["clips", "annotations"]
    process_params:
      max_clips: 10
```

### Command Line Usage

```bash
# Process any dataset
python -m data_manager.data_manager --config configs/my_datasets.yml
```

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/my_datasets.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)
```

## Dataset Selection Guide

### For Meeting Diarization
- **AMI**: English meetings, multiple mic types
- **ICSI**: Natural English meetings
- **AISHELL-4**: Mandarin meetings

### For Overlapped Speech
- **LibriheavyMix**: Synthetic overlaps, large scale
- **AMI/ICSI**: Natural overlaps in meetings

### For Evaluation
- **VoxConverse**: Standard diarization benchmark
- **Earnings-21**: Financial domain benchmark

### For Multi-modal Research
- **Ego4D**: First-person audio-visual
- **AVA-AVD**: Active speaker detection
- **MSDWild**: Multi-modal diarization

### For Domain-Specific Tasks
- **Earnings-21**: Financial domain
- **AISHELL-4**: Mandarin Chinese
- **Ego4D**: Egocentric scenarios

## Common Parameters

### Download Parameters

All datasets support these common download parameters:

- `target_dir`: Directory to download dataset to
- `force_download`: Force re-download even if present

### Process Parameters

All datasets support these common process parameters:

- `corpus_dir`: Path to downloaded dataset
- `output_dir`: Path to output manifests

## File Formats

### Input Formats
- **Audio**: WAV, MP3, MP4 (video), SPH
- **Annotations**: RTTM, JSON, JSONL, Lab, TextGrid

### Output Formats
All datasets generate Lhotse manifests:
- `*_recordings.jsonl.gz`: Recording metadata
- `*_supervisions.jsonl.gz`: Segment annotations
- `*_cuts.jsonl.gz`: Combined cuts (when applicable)

## Troubleshooting

### Common Issues

1. **Download Failures**: Check internet connection, verify URLs
2. **Disk Space**: Ensure sufficient space (some datasets are 100+ GB)
3. **Dependencies**: Install required system tools (ffmpeg, wget, etc.)
4. **Credentials**: Some datasets require registration (Ego4D, AVA-AVD)

### Getting Help

- Check individual dataset documentation for specific issues
- Review [Data Manager Overview](../../datasets.md)
- See [Configuration Guide](../../datasets.md#configuration)

## Contributing

To add a new dataset recipe:

1. Create recipe in `data_manager/recipes/`
2. Add parameter classes in `data_manager/dataset_types.py`
3. Update parameter mappings in `data_manager/parse_args.py`
4. Create documentation in this directory
5. Add tests and examples

## License

Each dataset has its own license. Please review individual dataset documentation for license terms before use.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Dataset Types Reference](../../dataset_types.md)
- [Lhotse Documentation](https://lhotse.readthedocs.io/)

