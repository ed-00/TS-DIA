# Dataset Parameters Comprehensive Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Global Configuration](#global-configuration)
3. [Feature Extraction Parameters](#feature-extraction-parameters)
4. [Dataset Parameters Reference](#dataset-parameters-reference)
5. [Common Patterns](#common-patterns)
6. [Integration Examples](#integration-examples)

---

## Introduction

This guide provides a comprehensive reference for all dataset parameters supported by the TS-DIA data manager. It covers:

- **Global configuration** settings that apply to all datasets
- **Feature extraction** parameters for fbank, MFCC, and spectrogram features
- **Dataset-specific** download and processing parameters for 50+ datasets
- **Integration examples** showing how to use these configurations in training

### Purpose

Each dataset has a corresponding YAML configuration file in `configs/datasets/` that you can copy and paste into your training configuration. All parameters are validated against the codebase and include proper type annotations and default values.

### How to Use

1. Browse the [Dataset Parameters Reference](#dataset-parameters-reference) to find your dataset
2. Copy the corresponding YAML file from `configs/datasets/{dataset_name}.yml`
3. Paste into your training configuration
4. Customize parameters as needed
5. Override global settings per dataset if required

---

## Global Configuration

The `global_config` section defines settings that apply to all datasets in your pipeline.

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | `./data` | Base directory for dataset downloads |
| `output_dir` | str | `./manifests` | Base directory for manifest outputs |
| `force_download` | bool | `false` | Force re-download even if dataset exists |

### Directory Structure

The system automatically creates organized directory structures:

```
{corpus_dir}/
├── {dataset_name}/          # Dataset-specific subdirectories
└── ...

{output_dir}/
├── {dataset_name}/          # Dataset-specific manifest directories
│   ├── {dataset}_recordings_{split}.jsonl.gz
│   ├── {dataset}_supervisions_{split}.jsonl.gz
│   └── ...
└── ...
```

---

## Feature Extraction Parameters

Feature extraction is configured globally and applies to all datasets. The TS-DIA system supports three feature types: **fbank** (filter bank), **MFCC**, and **spectrogram**.

### Feature Type Selection

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `feature_type` | str | `fbank` | `fbank`, `mfcc`, `spectrogram` | Type of acoustic features to extract |

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_rate` | int | `16000` | Target sampling rate in Hz |
| `frame_length` | float | `0.025` | Frame/window length in seconds (25ms) |
| `frame_shift` | float | `0.01` | Frame shift/hop length in seconds (10ms) |

### Window and FFT Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `round_to_power_of_two` | bool | `true` | Round window size to nearest power of 2 |
| `remove_dc_offset` | bool | `true` | Remove DC offset before processing |
| `preemph_coeff` | float | `0.97` | Pre-emphasis coefficient (typically 0.97) |
| `window_type` | str | `povey` | Window function: `povey`, `hanning`, `hamming`, `blackman` |
| `dither` | float | `0.0` | Dithering factor for numerical stability |
| `snip_edges` | bool | `false` | Whether to snip edges during feature extraction |

### Energy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `energy_floor` | float | `1e-10` | Floor value for energy computation |
| `raw_energy` | bool | `true` | Use raw energy (before windowing) |
| `use_energy` | bool | `false` | Use energy instead of C0 for MFCC |
| `use_fft_mag` | bool | `false` | Use FFT magnitude instead of power |

### Mel Filterbank Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `low_freq` | float | `20.0` | Lower frequency bound in Hz |
| `high_freq` | float | `-400.0` | Upper frequency bound (-400 means nyquist - 400) |
| `num_filters` | int | `23` | Number of triangular mel filters (for spectrogram) |
| `torchaudio_compatible_mel_scale` | bool | `true` | Use torchaudio-compatible mel scale |
| `num_mel_bins` | int | `80` | Number of mel-frequency bins (for fbank/mfcc) |
| `norm_filters` | bool | `false` | Normalize mel filter weights |

### MFCC-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_ceps` | int | `13` | Number of cepstral coefficients |
| `cepstral_lifter` | int | `22` | Cepstral liftering coefficient |

### Storage and Computation Parameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `storage_path` | str/null | `null` | Path or null | Path to store features (null = in-memory) |
| `num_jobs` | int | `1` | 1-N | Number of parallel jobs for feature extraction |
| `storage_type` | str | `lilcom_chunky` | `lilcom_chunky`, `lilcom_files`, `numpy`, `hdf5` | Storage format for features |
| `mix_eagerly` | bool | `true` | - | Whether to mix cuts eagerly during extraction |
| `progress_bar` | bool | `true` | - | Show progress bar during feature extraction |
| `device` | str | `cpu` | `cpu`, `cuda` | Device for computation |

### Feature Extraction Use Cases

#### Fbank (Filter Bank) - Default for Diarization

```yaml
global_config:
  feature_type: fbank
  num_mel_bins: 80
  frame_length: 0.025
  frame_shift: 0.01
  sampling_rate: 16000
  low_freq: 20.0
  high_freq: -400.0
  dither: 0.0
```

**Best for:** Speaker diarization, speaker verification, most modern neural networks

#### MFCC (Mel-Frequency Cepstral Coefficients)

```yaml
global_config:
  feature_type: mfcc
  num_mel_bins: 40
  num_ceps: 13
  frame_length: 0.025
  frame_shift: 0.01
  sampling_rate: 16000
  use_energy: true
  cepstral_lifter: 22
```

**Best for:** Traditional ASR systems, GMM-based models, classical speech processing

#### Spectrogram

```yaml
global_config:
  feature_type: spectrogram
  num_filters: 80
  frame_length: 0.025
  frame_shift: 0.01
  sampling_rate: 16000
```

**Best for:** End-to-end models, raw spectral analysis

### Storage Type Options

#### lilcom_chunky (Default - Recommended)

- Lossy compression optimized for speech
- Efficient storage and fast loading
- Good balance of size and quality

#### lilcom_files

- Lossy compression, one file per feature
- More flexible but more files

#### numpy

- Lossless storage as numpy arrays
- Larger file sizes
- Easy to work with

#### hdf5

- Lossless storage in HDF5 format
- Good for large datasets
- Supports compression

### Performance Tuning

#### CPU-Intensive Processing

```yaml
global_config:
  num_jobs: 8          # Use 8 parallel workers
  device: cpu
  progress_bar: true
```

#### GPU Acceleration

```yaml
global_config:
  num_jobs: 1          # GPU typically single-threaded
  device: cuda
  storage_type: lilcom_chunky
```

#### Large-Scale Processing

```yaml
global_config:
  storage_path: ./features  # Store to disk
  storage_type: lilcom_chunky
  num_jobs: 16
  progress_bar: true
```

---

## Dataset Parameters Reference

This section provides a comprehensive reference for all 50+ supported datasets. For detailed examples, see the individual YAML files in `configs/datasets/`.

### Dataset Categories

- [Diarization Datasets](#diarization-datasets)
- [ASR Datasets](#asr-datasets)
- [TTS Datasets](#tts-datasets)
- [Meeting Datasets](#meeting-datasets)
- [Augmentation Datasets](#augmentation-datasets)
- [Multi-language Datasets](#multi-language-datasets)

---

### Diarization Datasets

#### AMI Meeting Corpus

**File:** `configs/datasets/ami.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `annotations` | str/null | `null` | Path to annotations (auto-downloaded if null) |
| `url` | str | `http://groups.inf.ed.ac.uk/ami` | Base URL for downloads |
| `mic` | str | `ihm` | Microphone type: `ihm`, `sdm`, `mdm` |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `data_dir` | str/null | `null` | Alternative to corpus_dir |
| `annotations_dir` | str/null | `null` | Path to annotations directory |
| `mic` | str | `ihm` | Microphone type to process |
| `partition` | str | `full-corpus` | Partition: `full-corpus`, `full-corpus-asr`, `scenario-only` |
| `normalize_text` | str | `kaldi` | Text normalization: `none`, `upper`, `kaldi` |
| `max_words_per_segment` | int/null | `null` | Split long segments |
| `merge_consecutive` | bool | `false` | Merge consecutive same-speaker segments |

---

#### VoxConverse

**File:** `configs/datasets/voxconverse.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `corpus_dir` | str/null | `null` | Corpus directory path |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str/null | `null` | Path to output manifests |
| `split_test` | bool | `false` | Split test set |

---

#### ICSI Meeting Corpus

**File:** `configs/datasets/icsi.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `audio_dir` | str/null | `null` | Path to audio directory |
| `transcripts_dir` | str/null | `null` | Path to transcripts |
| `url` | str | `http://groups.inf.ed.ac.uk/ami` | Base URL |
| `mic` | str | `ihm` | Microphone type: `ihm`, `mdm` |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `audio_dir` | str/null | `null` | Path to audio directory |
| `transcripts_dir` | str/null | `null` | Path to transcripts |
| `mic` | str | `ihm` | Microphone type |
| `normalize_text` | str | `kaldi` | Text normalization |
| `save_to_wav` | bool | `false` | Convert SPH files to WAV |

---

#### Ego4D Audio Diarization

**File:** `configs/datasets/ego4d.yml`

**Environment Variables Required:**
- `AWS_ACCESS_KEY_ID`: AWS access key from Ego4D registration
- `AWS_SECRET_ACCESS_KEY`: AWS secret key from Ego4D registration

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `dataset_parts` | list/null | `null` | Parts to download: `clips`, `annotations` |
| `install_cli` | bool | `true` | Auto-install Ego4D CLI |
| `timeout` | int | `3600` | Download timeout in seconds |
| `env_file` | str/null | `null` | Path to .env file |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `extract_audio` | bool | `true` | Extract audio from video files |
| `audio_sample_rate` | int | `16000` | Sample rate for extracted audio |
| `min_segment_duration` | float | `0.5` | Minimum segment duration in seconds |
| `max_segment_duration` | float | `30.0` | Maximum segment duration in seconds |
| `max_clips` | int | `0` | Limit number of clips (0 = no limit) |
| `annotation_subset` | str/null | `null` | Filter annotations by type (e.g., "av") |

---

#### MSDWILD

**File:** `configs/datasets/mswild.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `download_audio` | bool | `true` | Download audio files (~7.56 GB) |
| `download_video` | bool | `false` | Download video files (~43.14 GB) |
| `download_faces` | bool | `false` | Download cropped faces (~14.49 GB) |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str/null | `null` | Path to output manifests |
| `splits` | dict/null | `null` | Split mapping: `{"train": "few_train", "dev": "few_val", "test": "many_val"}` |

---

#### AVA-AVD (Active Speaker Detection)

**File:** `configs/datasets/ava_avd.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `download_annotations` | bool | `true` | Download annotation files |
| `download_videos` | bool | `true` | Download video files |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str/null | `null` | Path to output manifests |
| `splits` | dict/null | `null` | Split mapping: `{"train": "train", "val": "val", "test": "test"}` |

---

#### LibriheavyMix (Overlapped Speech)

**File:** `configs/datasets/libriheavy_mix.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `dataset_parts` | str/list | `small` | Dataset splits: `small`, `medium`, `large`, `dev`, `test` |
| `speaker_counts` | int/list | `[1, 2, 3, 4]` | Number of speakers per mixture |
| `cache_dir` | str/null | `null` | HuggingFace cache directory |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str/null | `null` | Path to output manifests |
| `dataset_parts` | str/list | `small` | Dataset parts to process |
| `speaker_counts` | int/list | `[1, 2, 3, 4]` | Speaker counts to process |
| `splits` | dict/null | `null` | Custom split mapping |
| `min_speakers` | int | `1` | Minimum speakers to include |
| `max_speakers` | int | `4` | Maximum speakers to include |

---

### ASR Datasets

#### LibriSpeech

**File:** `configs/datasets/librispeech.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `dataset_parts` | str/list | `mini_librispeech` | Parts to download |
| `alignments` | bool | `false` | Download alignments |
| `base_url` | str | `http://www.openslr.org/resources` | Base URL |
| `alignments_url` | str/null | `null` | Alignments URL |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `alignments_dir` | str/null | `null` | Path to alignments |
| `dataset_parts` | str/list | `auto` | Parts to process |
| `normalize_text` | str | `none` | Text normalization: `none`, `upper`, `lower`, `kaldi` |

---

#### TIMIT

**File:** `configs/datasets/timit.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `base_url` | str | `http://www.openslr.org/resources` | Base URL |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `num_phones` | int | `48` | Number of phoneme classes (48 or 39) |

---

#### Earnings-21

**File:** `configs/datasets/earnings21.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `url` | str/null | `null` | Download URL (auto-detected) |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `normalize_text` | bool | `false` | Normalize text |

---

#### AISHELL-4 (Mandarin)

**File:** `configs/datasets/aishell4.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `base_url` | str | `http://www.openslr.org/resources` | Base URL |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `normalize_text` | bool | `false` | Normalize Chinese text |

---

### TTS Datasets

#### LJSpeech

**File:** `configs/datasets/ljspeech.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str/null | `null` | Path to output manifests |

---

#### LibriTTS

**File:** `configs/datasets/libritts.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `use_librittsr` | bool | `false` | Use LibriTTS-R variant |
| `dataset_parts` | str/list | `all` | Parts to download |
| `base_url` | str | `http://www.openslr.org/resources` | Base URL |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `dataset_parts` | str/list | `all` | Parts to process |
| `link_previous_utt` | bool | `false` | Link to previous utterance |

---

#### VCTK

**File:** `configs/datasets/vctk.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `use_edinburgh_vctk_url` | bool | `false` | Use Edinburgh server URL |
| `url` | str/null | `null` | Custom URL |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `use_edinburgh_vctk_url` | bool | `false` | Use Edinburgh server |
| `mic_id` | str | `mic2` | Microphone ID to use |

---

### Augmentation Datasets

#### MUSAN (Background Noise)

**File:** `configs/datasets/musan.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `url` | str/null | `null` | Download URL |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `parts` | list | `["music", "speech", "noise"]` | Parts to process |
| `use_vocals` | bool | `true` | Include vocal tracks |

---

#### RIR Noise (Room Impulse Response)

**File:** `configs/datasets/rir_noise.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `url` | str/null | `null` | Download URL |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `parts` | list | `["point_noise", "iso_noise", "real_rir", "sim_rir"]` | Parts to process |

---

### Multi-language Datasets

#### FLEURS

**File:** `configs/datasets/fleurs.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `languages` | str/list | `all` | Languages to download |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `languages` | str/list | `all` | Languages to process |

---

#### mTEDx

**File:** `configs/datasets/mtedx.yml`

**Download Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_dir` | str | Required | Directory to download to |
| `force_download` | bool | `false` | Force re-download |
| `languages` | str/list | `all` | Languages to download |

**Process Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_dir` | str | Required | Path to downloaded corpus |
| `output_dir` | str | Required | Path to output manifests |
| `languages` | str/list | `all` | Languages to process |

---

### Complete Dataset List

For a complete list of all 50+ supported datasets with individual YAML configuration files, see `configs/datasets/`:

**Diarization & Meeting:**
- ami, icsi, voxconverse, ego4d, mswild, ava_avd, dipco, chime6, libricss

**ASR (English):**
- librispeech, libritts, tedlium, earnings21, earnings22, spgispeech, peoples_speech, this_american_life

**ASR (Mandarin):**
- aishell, aishell3, aishell4, magicdata, stcmds, thchs_30, primewords

**ASR (Multi-language):**
- fleurs, mtedx, voxpopuli, gigast

**TTS:**
- ljspeech, libritts, vctk, baker_zh, hifitts, daily_talk

**Speaker Recognition:**
- voxceleb1, voxceleb2

**Augmentation:**
- musan, rir_noise, but_reverb_db

**Other:**
- timit, yesno, speechcommands, grid, and more

---

## Common Patterns

### Microphone Types

Many meeting datasets support multiple microphone types:

| Type | Description | Quality | Use Case |
|------|-------------|---------|----------|
| `ihm` | Individual Headset Microphone | Best | Clean diarization, close-talk |
| `sdm` | Single Distant Microphone | Good | Single-channel far-field |
| `mdm` | Multiple Distant Microphones | Variable | Multi-channel far-field |

**Example:**

```yaml
datasets:
  - name: ami
    download_params:
      mic: ihm
    process_params:
      mic: ihm
```

### Text Normalization

Text normalization options vary by dataset:

| Option | Description | Example |
|--------|-------------|---------|
| `none` | No normalization | Original text |
| `upper` | Uppercase | "HELLO WORLD" |
| `lower` | Lowercase | "hello world" |
| `kaldi` | Kaldi-style normalization | Removes punctuation, etc. |

**Example:**

```yaml
datasets:
  - name: librispeech
    process_params:
      normalize_text: lower
```

### Dataset Parts Selection

Multi-part datasets allow selecting specific subsets:

**Example:**

```yaml
datasets:
  - name: librispeech
    download_params:
      dataset_parts: ["train-clean-100", "dev-clean", "test-clean"]
    process_params:
      dataset_parts: ["train-clean-100", "dev-clean", "test-clean"]
```

### Multi-language Datasets

For multi-language datasets, specify languages:

**Example:**

```yaml
datasets:
  - name: fleurs
    download_params:
      languages: ["en_us", "es_419", "fr_fr"]
    process_params:
      languages: ["en_us", "es_419", "fr_fr"]
```

---

## Integration Examples

### Basic Integration

Copy a dataset config into your training configuration:

```yaml
# From configs/datasets/voxconverse.yml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false
  
  # Feature extraction
  feature_type: fbank
  num_mel_bins: 80
  frame_length: 0.025
  frame_shift: 0.01
  sampling_rate: 16000

datasets:
  - name: voxconverse

# Your training configuration
training:
  epochs: 10
  batch_size: 32
  # ... other training params
```

### Multi-Dataset Training

Combine multiple datasets with shared global config:

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false
  
  # Shared feature extraction
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000

datasets:
  - name: voxconverse
  
  - name: ami
    download_params:
      mic: ihm
    process_params:
      mic: ihm
      partition: full-corpus
  
  - name: icsi
    process_params:
      mic: ihm
```

### Override Global Settings Per Dataset

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false
  
  # Default fbank config
  feature_type: fbank
  num_mel_bins: 80

datasets:
  # Use global fbank config
  - name: voxconverse
  
  # Override for MFCC
  - name: timit
    # Note: Feature config is global, but you can create separate configs
    process_params:
      num_phones: 48
```

### Custom Output Directories

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  
datasets:
  - name: ami
    process_params:
      output_dir: ./manifests/ami_custom  # Override global
```

### Large-Scale Training Setup

```yaml
global_config:
  corpus_dir: /mnt/datasets
  output_dir: /mnt/manifests
  force_download: false
  
  # High-performance feature extraction
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000
  storage_type: lilcom_chunky
  storage_path: /mnt/features
  num_jobs: 32
  device: cpu

datasets:
  - name: libriheavy_mix
    download_params:
      dataset_parts: ["medium", "large"]
      speaker_counts: [2, 3, 4]
    process_params:
      dataset_parts: ["medium", "large"]
      speaker_counts: [2, 3, 4]
  
  - name: ami
    download_params:
      mic: mdm
    process_params:
      mic: mdm
      partition: full-corpus
```

---

## Quick Reference

### File Locations

- **Individual Dataset YAMLs:** `configs/datasets/{dataset_name}.yml`
- **This Guide:** `docs/datasets/PARAMETERS_GUIDE.md`
- **Dataset Documentation:** `docs/datasets/recipes/{dataset_name}.md`
- **Source Code:** `data_manager/dataset_types.py`

### Getting Started

1. Browse available datasets in `configs/datasets/`
2. Copy the YAML file you need
3. Paste into your training config
4. Adjust parameters as needed
5. Run your training pipeline

### Support

For detailed information on specific datasets:
- See individual markdown docs in `docs/datasets/recipes/`
- Check Lhotse documentation for Lhotse-based datasets
- Review source code in `data_manager/recipes/` for custom recipes

---

## License

Copyright (c) 2025 Abed Hameed. Licensed under Apache 2.0 license.

