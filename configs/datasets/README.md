# Dataset Configuration Files

This directory contains ready-to-use YAML configuration files for all 50+ datasets supported by TS-DIA.

## Quick Start

1. **Find your dataset** in the list below
2. **Copy the YAML file** you need
3. **Paste into your training config**
4. **Customize parameters** as needed

## Complete Reference

For detailed parameter documentation, see:
- **[Comprehensive Parameters Guide](../../docs/datasets/PARAMETERS_GUIDE.md)**
- **[Dataset Recipes Documentation](../../docs/datasets/recipes/)**

## Dataset Categories

### Diarization & Meeting Datasets

| Dataset | File | Description | Size |
|---------|------|-------------|------|
| AMI | [ami.yml](ami.yml) | Meeting recordings, multiple mic types | ~100 hours |
| ICSI | [icsi.yml](icsi.yml) | Natural meeting recordings | ~72 hours |
| VoxConverse | [voxconverse.yml](voxconverse.yml) | Speaker diarization benchmark | ~32 hours |
| Ego4D | [ego4d.yml](ego4d.yml) | Egocentric audio diarization | Variable |
| MSDWILD | [mswild.yml](mswild.yml) | Multi-modal speaker diarization | ~7.56 GB |
| AVA-AVD | [ava_avd.yml](ava_avd.yml) | Active speaker detection | Variable |
| LibriheavyMix | [libriheavy_mix.yml](libriheavy_mix.yml) | Overlapped speech (1-4 speakers) | 100h-20,000h |
| DIPCO | [dipco.yml](dipco.yml) | Dinner party conversational speech | Variable |
| LibriCSS | [libricss.yml](libricss.yml) | Continuous speech separation | Variable |
| AliMeeting | [ali_meeting.yml](ali_meeting.yml) | Mandarin meeting corpus | Variable |

### ASR Datasets (English)

| Dataset | File | Description | Size |
|---------|------|-------------|------|
| LibriSpeech | [librispeech.yml](librispeech.yml) | Large-scale English ASR | 100-1000 hours |
| TIMIT | [timit.yml](timit.yml) | Phone recognition corpus | ~5 hours |
| TEDLium | [tedlium.yml](tedlium.yml) | TED talk recordings | ~450 hours |
| Earnings-21 | [earnings21.yml](earnings21.yml) | Financial earnings calls | ~39 hours |
| Earnings-22 | [earnings22.yml](earnings22.yml) | Extended earnings calls | Variable |
| SPGISpeech | [spgispeech.yml](spgispeech.yml) | Financial domain corpus | ~5,000 hours |
| People's Speech | [peoples_speech.yml](peoples_speech.yml) | Multi-domain English corpus | ~30,000 hours |
| This American Life | [this_american_life.yml](this_american_life.yml) | Podcast episodes | Variable |

### ASR Datasets (Mandarin)

| Dataset | File | Description | Size |
|---------|------|-------------|------|
| AISHELL | [aishell.yml](aishell.yml) | Mandarin speech corpus | ~170 hours |
| AISHELL-3 | [aishell3.yml](aishell3.yml) | Multi-speaker Mandarin TTS | ~85 hours |
| AISHELL-4 | [aishell4.yml](aishell4.yml) | Mandarin meeting corpus | ~120 hours |
| MagicData | [magicdata.yml](magicdata.yml) | Mandarin read speech | ~755 hours |
| STCMDS | [stcmds.yml](stcmds.yml) | Mandarin speech commands | ~100 hours |
| THCHS-30 | [thchs_30.yml](thchs_30.yml) | Tsinghua Mandarin corpus | ~30 hours |
| Primewords | [primewords.yml](primewords.yml) | Wake word detection | ~260 hours |
| Mobvoi Hotwords | [mobvoihotwords.yml](mobvoihotwords.yml) | Chinese wake words | Variable |

### TTS Datasets

| Dataset | File | Description | Size |
|---------|------|-------------|------|
| LJSpeech | [ljspeech.yml](ljspeech.yml) | Single speaker (female) TTS | ~24 hours |
| LibriTTS | [libritts.yml](libritts.yml) | Multi-speaker TTS | ~585 hours |
| LibriTTS-R | [librittsr.yml](librittsr.yml) | Cleaned LibriTTS | ~585 hours |
| VCTK | [vctk.yml](vctk.yml) | Multi-speaker, various accents | ~44 hours |
| Baker (Mandarin) | [baker_zh.yml](baker_zh.yml) | Mandarin female speaker | ~12 hours |
| Hi-Fi TTS | [hifitts.yml](hifitts.yml) | High-fidelity multi-speaker | ~291 hours |
| DailyTalk | [daily_talk.yml](daily_talk.yml) | Conversational speech | Variable |
| CMU Arctic | [cmu_arctic.yml](cmu_arctic.yml) | Multi-speaker with accents | ~18 hours |
| CMU Indic | [cmu_indic.yml](cmu_indic.yml) | Indian language TTS | Variable |
| BVCC | [bvcc.yml](bvcc.yml) | Voice conversion challenge | Variable |

### Multi-language Datasets

| Dataset | File | Description | Size |
|---------|------|-------------|------|
| FLEURS | [fleurs.yml](fleurs.yml) | 102 languages | Variable |
| mTEDx | [mtedx.yml](mtedx.yml) | Multilingual TED talks | Variable |
| VoxPopuli | [voxpopuli.yml](voxpopuli.yml) | European Parliament (15 langs) | ~1,800 hours |
| GigaST | [gigast.yml](gigast.yml) | Speech translation corpus | ~10,000 hours |
| ReazonSpeech | [reazonspeech.yml](reazonspeech.yml) | Japanese ASR | ~35,000 hours |
| HEROICO | [heroico.yml](heroico.yml) | Spanish read speech | ~14 hours |
| HI-MIA | [himia.yml](himia.yml) | Hindi meeting corpus | Variable |
| XBMU-AMDO31 | [xbmu_amdo31.yml](xbmu_amdo31.yml) | Tibetan speech | Variable |

### Speaker Recognition

| Dataset | File | Description | Size |
|---------|------|-------------|------|
| VoxCeleb1 | [voxceleb1.yml](voxceleb1.yml) | Speaker identification | ~352 hours |
| VoxCeleb2 | [voxceleb2.yml](voxceleb2.yml) | Extended speaker ID | ~2,442 hours |

### Augmentation & Noise

| Dataset | File | Description | Size |
|---------|------|-------------|------|
| MUSAN | [musan.yml](musan.yml) | Music, speech, noise | ~109 hours |
| RIR Noise | [rir_noise.yml](rir_noise.yml) | Room impulse responses | Variable |
| BUT ReverbDB | [but_reverb_db.yml](but_reverb_db.yml) | Reverberation database | Variable |

### Specialized Datasets

| Dataset | File | Description | Size |
|---------|------|-------------|------|
| CHiME-6 | [chime6.yml](chime6.yml) | Far-field conversational ASR | ~40 hours |
| Speech Commands | [speechcommands.yml](speechcommands.yml) | Keyword spotting | ~105,000 utterances |
| YesNo | [yesno.yml](yesno.yml) | Hebrew binary classification | ~60 utterances |
| GRID | [grid.yml](grid.yml) | Audio-visual speech | Variable |
| LibriMix | [librimix.yml](librimix.yml) | Speech separation (2 speakers) | Variable |
| ATCOSIM | [atcosim.yml](atcosim.yml) | Air traffic control | ~10 hours |
| UWB-ATCC | [uwb_atcc.yml](uwb_atcc.yml) | Air traffic communications | ~5 hours |
| EARS | [ears.yml](ears.yml) | Lhotse dataset | Variable |
| EDACC | [edacc.yml](edacc.yml) | Dysarthric speech | Variable |
| MDCC | [mdcc.yml](mdcc.yml) | Multi-domain chatbot | Variable |
| Medical | [medical.yml](medical.yml) | Medical domain speech | Variable |
| NSC | [nsc.yml](nsc.yml) | Singapore English | Variable |
| ADEPT | [adept.yml](adept.yml) | Lhotse dataset | Variable |

## Usage Examples

### Single Dataset

```yaml
# Copy from any dataset YAML file above
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  force_download: false
  
  # Feature extraction
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000

datasets:
  - name: voxconverse
```

### Multiple Datasets

Combine multiple dataset configs into one file:

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  
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
  
  - name: libriheavy_mix
    download_params:
      dataset_parts: small
      speaker_counts: [2, 3, 4]
```

### Override Feature Extraction Per Dataset

Note: Feature extraction is global. To use different features, create separate config files.

```yaml
# Config 1: fbank_datasets.yml
global_config:
  feature_type: fbank
  num_mel_bins: 80

datasets:
  - name: voxconverse
  - name: ami

# Config 2: mfcc_datasets.yml
global_config:
  feature_type: mfcc
  num_ceps: 13

datasets:
  - name: timit
```

## Parameter Validation

All parameters in these YAML files are validated against the source code in:
- `data_manager/dataset_types.py` - Parameter definitions
- `data_manager/parse_args.py` - Parameter mapping

## Feature Extraction

Feature extraction is configured globally. Key parameters:

- **feature_type**: `fbank` (default), `mfcc`, or `spectrogram`
- **num_mel_bins**: Number of mel bins (80 for diarization)
- **sampling_rate**: Target sampling rate (16000 Hz typical)
- **storage_type**: `lilcom_chunky`, `lilcom_files`, `numpy`, `hdf5`
- **device**: `cpu` or `cuda`

For complete feature extraction documentation, see the [Parameters Guide](../../docs/datasets/PARAMETERS_GUIDE.md#feature-extraction-parameters).

## Getting Help

- **[Comprehensive Parameters Guide](../../docs/datasets/PARAMETERS_GUIDE.md)** - Complete parameter reference
- **[Dataset Recipes](../../docs/datasets/recipes/)** - Individual dataset documentation
- **[Data Manager Overview](../../docs/datasets/README.md)** - System overview

## License

Each dataset has its own license. Review individual dataset documentation before use.

