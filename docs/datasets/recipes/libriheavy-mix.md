# LibriheavyMix: Large-Scale Overlapped Speech Dataset

## Overview

LibriheavyMix is a 20,000-hour synthesized corpus for overlapped speech separation and diarization. The dataset contains mixtures with 1-4 speakers and includes reverberation effects, making it ideal for training and evaluating speaker diarization systems on challenging overlapped speech scenarios.

## Features

- **Large Scale**: Up to 20,000 hours of synthesized mixtures
- **Multiple Splits**: Small (100h), Medium (900h), Large (9000h), Dev, Test
- **Speaker Overlap**: 1-4 speakers per mixture
- **Reverberation**: Includes room impulse response (RIR) effects
- **SNR Augmentation**: Various signal-to-noise ratio conditions
- **HuggingFace Hosting**: Easy download from HuggingFace datasets

## Dataset Information

- **Name**: `libriheavy_mix`
- **Type**: Overlapped speech diarization
- **License**: Check HuggingFace dataset page for license
- **Homepage**: https://huggingface.co/datasets/zrjin/LibriheavyMix-small
- **Paper**: TBD (check HuggingFace for citation)
- **Splits**: small, medium, large, dev, test

## Prerequisites

### System Dependencies
- None (pure Python download and processing)

### Python Dependencies
- `lhotse` (for manifest generation and processing)
- Standard library only for download

## Download Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str/Path | **Required** | Directory to download dataset to |
| `force_download` | bool | `False` | Force re-download even if already present |
| `dataset_parts` | str/List[str] | `"small"` | Dataset splits to download |
| `speaker_counts` | int/List[int] | `[1, 2, 3, 4]` | Number of speakers per mixture |
| `cache_dir` | str/Path | `None` | HuggingFace cache directory (optional) |

### Available Dataset Parts

- **small**: 100 hours (good for quick experiments)
- **medium**: 900 hours (medium-scale training)
- **large**: 9,000 hours (large-scale training)
- **dev**: Development set
- **test**: Test set

### Speaker Counts

- **1**: Single speaker (no overlap)
- **2**: Two-speaker mixtures
- **3**: Three-speaker mixtures
- **4**: Four-speaker mixtures

## Process Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str/Path | **Required** | Path to downloaded LibriheavyMix dataset |
| `output_dir` | str/Path | `None` | Path to output manifests directory |
| `dataset_parts` | str/List[str] | `"small"` | Dataset parts to process |
| `speaker_counts` | int/List[int] | `[1, 2, 3, 4]` | Speaker counts to process |
| `splits` | Dict[str, str] | `None` | Custom split mapping (optional) |
| `min_speakers` | int | `1` | Minimum number of speakers to include |
| `max_speakers` | int | `4` | Maximum number of speakers to include |

## Configuration Example

### Minimal Configuration (Small Dataset)

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: libriheavy_mix
```

### Full Configuration (Multiple Splits)

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false

datasets:
  - name: libriheavy_mix
    download_params:
      force_download: false
      dataset_parts: ["small", "dev", "test"]
      speaker_counts: [1, 2, 3, 4]
      cache_dir: null
    process_params:
      dataset_parts: ["small", "dev", "test"]
      speaker_counts: [1, 2, 3, 4]
      min_speakers: 1
      max_speakers: 4
```

### Two-Speaker Overlap Only

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: libriheavy_mix
    download_params:
      dataset_parts: "small"
      speaker_counts: 2
    process_params:
      speaker_counts: 2
      min_speakers: 2
      max_speakers: 2
```

### Large-Scale Training Setup

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: libriheavy_mix
    download_params:
      dataset_parts: ["medium", "large"]
      speaker_counts: [2, 3, 4]  # Skip single speaker
    process_params:
      dataset_parts: ["medium", "large"]
      speaker_counts: [2, 3, 4]
      min_speakers: 2
      max_speakers: 4
```

## Output Structure

```
./data/libriheavy_mix/               # Downloaded dataset
├── small/                           # Small split (100h)
│   ├── 1spk/
│   │   ├── lsheavymix_cuts_dev_1spk.jsonl.gz
│   │   └── audio/
│   ├── 2spk/
│   │   ├── lsheavymix_cuts_dev_2spk_snr_aug_mono_rir.jsonl.gz
│   │   └── audio/
│   ├── 3spk/
│   └── 4spk/
├── medium/                          # Medium split (900h)
├── large/                           # Large split (9000h)
├── dev/                             # Development set
├── test/                            # Test set
└── .completed                       # Download completion marker

./manifests/libriheavy_mix/          # Generated manifests
├── libriheavy_mix_cuts_small_1spk.jsonl.gz
├── libriheavy_mix_cuts_small_2spk.jsonl.gz
├── libriheavy_mix_cuts_small_3spk.jsonl.gz
├── libriheavy_mix_cuts_small_4spk.jsonl.gz
└── ...
```

## Usage Example

### Python API

```python
from data_manager import DatasetManager, parse_dataset_configs

# Load configuration
configs = parse_dataset_configs('configs/libriheavy_mix_config.yml')

# Download and process
cut_sets = DatasetManager.load_datasets(datasets=configs)

# Use for diarization training
for cut_set in cut_sets:
    print(f"Loaded {len(cut_set)} cuts")
    print(f"Total duration: {cut_set.duration / 3600:.2f} hours")
    cut_set.describe()
```

### Direct Recipe Usage

```python
from data_manager.recipes.libriheavy_mix import download_libriheavy_mix, prepare_libriheavy_mix

# Download small split with 2-speaker mixtures
corpus_dir = download_libriheavy_mix(
    target_dir="./data",
    dataset_parts="small",
    speaker_counts=[2, 3]
)

# Process
manifests = prepare_libriheavy_mix(
    corpus_dir=corpus_dir,
    output_dir="./manifests/libriheavy_mix",
    dataset_parts="small",
    speaker_counts=[2, 3]
)
```

### Command Line

```bash
# Process LibriheavyMix dataset
python -m data_manager.data_manager --config configs/libriheavy_mix_config.yml
```

## File Variants

LibriheavyMix provides multiple processing variants (in order of preference):

1. **`_snr_aug_mono_rir_fixed.jsonl.gz`**: SNR augmented + RIR + fixes
2. **`_snr_aug_mono_rir.jsonl.gz`**: SNR augmented + RIR
3. **`_snr_aug_mono.jsonl.gz`**: SNR augmented only
4. **`_base.jsonl.gz`**: Base mixtures

The download function automatically tries variants in order and uses the first available.

## Dataset Statistics

### Split Sizes
- **Small**: ~100 hours
- **Medium**: ~900 hours  
- **Large**: ~9,000 hours
- **Dev**: Development set (varies)
- **Test**: Test set (varies)

### Speaker Distribution
- **1 speaker**: Single speaker baseline
- **2 speakers**: Most common overlap scenario
- **3 speakers**: Challenging overlap
- **4 speakers**: Very challenging overlap

## Troubleshooting

### HuggingFace Download Fails
```
Failed to download any variant for LibriheavyMix-small-2spk
```
**Solution**:
- Check internet connection
- Verify HuggingFace is accessible
- Try downloading one speaker count at a time
- Check HuggingFace dataset page for availability

### Large Download Size
```
Download taking very long...
```
**Solution**:
- Start with `dataset_parts: "small"` for testing
- Download overnight for medium/large splits
- Use `speaker_counts: [2]` to reduce download size
- Consider downloading splits separately

### Missing Audio Files
```
WARNING: Audio file not found
```
**Solution**: Ensure both manifest and audio tar.gz files downloaded successfully.

### Out of Disk Space
```
No space left on device
```
**Solution**:
- Small: ~10 GB
- Medium: ~90 GB
- Large: ~900 GB
- Plan storage accordingly

## Performance Considerations

- **Download Size**:
  - Small: ~10 GB per speaker count
  - Medium: ~90 GB per speaker count
  - Large: ~900 GB per speaker count
- **Processing Speed**: Fast (pre-processed Lhotse manifests)
- **Memory Usage**: Low (lazy loading)
- **Training**: Ideal for large-scale diarization model training

## Use Cases

### Speaker Diarization Training
Train models on realistic overlapped speech:
```python
# Use 2-4 speaker mixtures for training
dataset_parts: ["medium"]
speaker_counts: [2, 3, 4]
```

### Overlap Detection
Focus on multi-speaker scenarios:
```python
# Only overlapped speech
speaker_counts: [2, 3, 4]
min_speakers: 2
```

### Baseline Comparison
Include single-speaker for baseline:
```python
# All speaker counts for comprehensive evaluation
speaker_counts: [1, 2, 3, 4]
```

## Citation

```bibtex
# Check HuggingFace dataset page for official citation
# https://huggingface.co/datasets/zrjin/LibriheavyMix-small
```

## License

Check the HuggingFace dataset page for license terms.

## Related Documentation

- [Data Manager Overview](../../datasets.md)
- [Dataset Configuration Guide](../../datasets.md#configuration)
- [Custom Recipes](../../../recipes.md)
- [Lhotse Documentation](https://lhotse.readthedocs.io/)

