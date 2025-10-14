# Configuration Examples

This guide provides practical, real-world configuration examples for common use cases.

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Model Architecture Examples](#model-architecture-examples)
3. [Training Strategy Examples](#training-strategy-examples)
4. [Dataset Configuration Examples](#dataset-configuration-examples)
5. [Production Configurations](#production-configurations)
6. [Troubleshooting Configs](#troubleshooting-configs)

---

## Quick Start Examples

### Minimal Configuration

The absolute minimum to start training:

```yaml
# configs/minimal.yml
model:
  model_type: encoder
  name: minimal_model
  
  encoder:
    d_model: 256
    num_layers: 4
    num_heads: 8
    attention_type: softmax
    activation: GELU
    num_classes: 1

global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  feature_type: fbank
  num_mel_bins: 80

datasets:
  - name: yesno

training:
  epochs: 10
  batch_size: 32
  
  optimizer:
    type: adamw
    lr: 0.0001
  
  scheduler:
    type: cosine
  
  checkpoint:
    save_dir: ./checkpoints/minimal
    interval: 1000
  
  logging:
    interval: 100
    tensorboard: true
  
  performance:
    num_workers: 2
  
  eval_knobs:
    label_type: binary
```

**Usage**:
```bash
python train.py --config configs/minimal.yml
```

### Quick Experiment

Fast iteration for development:

```yaml
# configs/quick_experiment.yml
model:
  model_type: encoder
  name: quick_test
  
  global_config:
    dropout: 0.1
    batch_size: 8
    d_ff: 4
    device: cpu
  
  encoder:
    d_model: 64
    num_layers: 2
    num_heads: 4
    attention_type: softmax
    activation: GELU
    num_classes: 1

global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  feature_type: fbank
  num_mel_bins: 64
  sampling_rate: 8000

datasets:
  - name: yesno

training:
  epochs: 2
  batch_size: 4
  random_seed: 42
  max_steps: 50                    # Stop after 50 steps
  
  optimizer:
    type: adam
    lr: 0.001
  
  scheduler:
    type: constant
  
  checkpoint:
    save_dir: ./test_checkpoints/quick
    interval: 20
  
  validation:
    interval: 10
    batch_size: 4
  
  logging:
    interval: 5
    tensorboard: false
    wandb: false
  
  performance:
    num_workers: 0                 # Avoid multiprocessing overhead
  
  eval_knobs:
    label_type: binary
```

**Usage**:
```bash
# Quick validation test
python train.py --config configs/quick_experiment.yml
```

---

## Model Architecture Examples

### 1. Encoder-Only (Classification/Diarization)

For tasks like speaker diarization, text classification, sequence labeling:

```yaml
# configs/encoder_diarization.yml
model:
  model_type: encoder
  name: diarization_encoder
  
  global_config:
    dropout: 0.1
    batch_size: 32
    d_ff: 4
    device: cuda
  
  encoder:
    d_model: 512
    num_layers: 12
    num_heads: 8
    attention_type: linear         # Efficient for long sequences
    activation: GEGLU
    nb_features: 256
    use_rezero: false
    use_scalenorm: false
    feature_redraw_interval: 1000
    auto_check_redraw: true
    num_classes: 1                 # Binary diarization output

global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000

datasets:
  - name: ami

training:
  epochs: 50
  batch_size: 32
  
  optimizer:
    type: adamw
    lr: 0.0001
    weight_decay: 0.01
  
  scheduler:
    type: cosine
    min_lr: 0.000001
    warmup_steps: 1000
  
  loss:
    main: bce_with_logits          # Binary cross-entropy for diarization
  
  checkpoint:
    save_dir: ./checkpoints/diarization
    interval: 1000
    save_total_limit: 5
  
  validation:
    interval: 500
    batch_size: 64
  
  early_stopping:
    patience: 5
    metric: val_loss
  
  logging:
    interval: 50
    tensorboard: true
    wandb: true
    wandb_project: speaker-diarization
  
  eval_knobs:
    label_type: binary
```

### 2. Decoder-Only (Language Modeling)

For autoregressive tasks like language modeling:

```yaml
# configs/decoder_lm.yml
model:
  model_type: decoder
  name: language_model
  
  global_config:
    dropout: 0.1
    batch_size: 32
    d_ff: 4
    device: cuda
  
  decoder:
    d_model: 512
    num_layers: 12
    num_heads: 8
    attention_type: causal_linear  # Causal for autoregressive
    activation: SWIGLU
    nb_features: 256
    use_cross_attention: false     # No encoder for LM
    feature_redraw_interval: 1000
    auto_check_redraw: true

global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000

datasets:
  - name: librispeech
    download_params:
      dataset_parts: mini_librispeech

training:
  epochs: 100
  batch_size: 64
  
  optimizer:
    type: adamw
    lr: 0.0003
    weight_decay: 0.01
    betas: [0.9, 0.95]
  
  scheduler:
    type: cosine
    min_lr: 0.00003
    warmup_steps: 2000
  
  loss:
    main: cross_entropy
    label_smoothing: 0.1
  
  gradient_clipping: 1.0
  gradient_accumulation_steps: 4
  mixed_precision: true
  
  checkpoint:
    save_dir: ./checkpoints/language_model
    interval: 1000
  
  logging:
    interval: 100
    tensorboard: true
  
  performance:
    num_workers: 4
    pin_memory: true
```

### 3. Encoder-Decoder (Sequence-to-Sequence)

For tasks like translation, summarization, speech recognition:

```yaml
# configs/encoder_decoder_translation.yml
model:
  model_type: encoder_decoder
  name: speech_translator
  
  global_config:
    dropout: 0.1
    batch_size: 32
    d_ff: 4
    device: cuda
  
  encoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    attention_type: linear
    activation: GEGLU
    nb_features: 256
  
  decoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    attention_type: causal_linear
    activation: SWIGLU
    nb_features: 256
    use_cross_attention: true      # Connect to encoder

global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000
  frame_length: 0.025
  frame_shift: 0.01

datasets:
  - name: librispeech
    download_params:
      dataset_parts: mini_librispeech

training:
  epochs: 100
  batch_size: 32
  
  optimizer:
    type: adamw
    lr: 0.0002
    weight_decay: 0.01
  
  scheduler:
    type: cosine
    min_lr: 0.00001
    warmup_steps: 4000
  
  loss:
    main: cross_entropy
    label_smoothing: 0.1
  
  gradient_clipping: 1.0
  mixed_precision: true
  
  checkpoint:
    save_dir: ./checkpoints/translator
    interval: 2000
    save_total_limit: 3
  
  validation:
    interval: 1000
    batch_size: 64
  
  logging:
    interval: 100
    tensorboard: true
    wandb: true
    wandb_project: speech-translation
  
  performance:
    num_workers: 8
    pin_memory: true
```

---

## Training Strategy Examples

### 1. Fast Prototyping

Small model, quick iterations:

```yaml
training:
  epochs: 10
  batch_size: 16
  max_steps: 1000               # Quick stop
  
  optimizer:
    type: adam
    lr: 0.001                   # Higher LR for faster convergence
  
  scheduler:
    type: constant              # No scheduling needed
  
  checkpoint:
    save_dir: ./checkpoints/prototype
    interval: 100               # Save often
  
  logging:
    interval: 10                # Frequent logging
    tensorboard: true
  
  performance:
    num_workers: 0              # Avoid overhead
```

### 2. Large-Scale Training

Production model with full optimization:

```yaml
training:
  epochs: 200
  batch_size: 256               # Large batch
  
  optimizer:
    type: adamw
    lr: 0.0001
    weight_decay: 0.01
    betas: [0.9, 0.999]
  
  scheduler:
    type: cosine
    min_lr: 0.000001
    warmup_steps: 10000         # Long warmup
  
  gradient_clipping: 1.0
  gradient_accumulation_steps: 8  # Effective batch = 256 * 8 = 2048
  mixed_precision: true
  
  feature_redraw_interval: 100
  
  loss:
    main: cross_entropy
    label_smoothing: 0.1
    auxiliary:
      norm_reg: 0.05            # Regularization
  
  validation:
    interval: 2000
    batch_size: 512
  
  early_stopping:
    patience: 10
    metric: val_loss
    min_delta: 0.0001
  
  checkpoint:
    save_dir: ./checkpoints/production
    interval: 5000
    save_total_limit: 5
    snapshot_optimizer: true
    snapshot_scheduler: true
  
  distributed:
    backend: nccl
    world_size: 8               # 8 GPUs
  
  logging:
    interval: 100
    tensorboard: true
    wandb: true
    wandb_project: production-model
    log_model: true
  
  performance:
    num_workers: 16
    pin_memory: true
    prefetch_factor: 4
    persistent_workers: true
```

### 3. Fine-Tuning

Resume from checkpoint with smaller LR:

```yaml
training:
  epochs: 20                    # Fewer epochs
  batch_size: 32
  
  optimizer:
    type: adamw
    lr: 0.00001                 # Much smaller LR
    weight_decay: 0.001         # Less regularization
  
  scheduler:
    type: linear                # Linear decay
    min_lr: 0.000001
    warmup_steps: 100           # Short warmup
  
  checkpoint:
    save_dir: ./checkpoints/finetuned
    interval: 500
    resume: ./checkpoints/pretrained/best_model.pt  # Resume from pretrained
  
  validation:
    interval: 200
  
  early_stopping:
    patience: 3                 # Stop early if overfitting
    metric: val_loss
```

### 4. Hyperparameter Search

Minimal config for hyperparameter tuning:

```yaml
training:
  epochs: 20
  batch_size: 32
  
  optimizer:
    type: adamw
    lr: 0.0001                  # Will be overridden by search
    weight_decay: 0.01
  
  scheduler:
    type: cosine
  
  checkpoint:
    save_dir: ./checkpoints/hp_search
    interval: 500
  
  early_stopping:
    patience: 3
    metric: val_loss
  
  tuning:
    library: raytune
    search_space:
      lr: [0.00001, 0.0001, 0.001]
      batch_size: [16, 32, 64]
      dropout: [0.1, 0.2, 0.3]
      weight_decay: [0.001, 0.01, 0.1]
```

---

## Dataset Configuration Examples

### 1. Single Dataset

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000

datasets:
  - name: ami
```

### 2. Multiple Datasets

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000

datasets:
  - name: ami
  - name: icsi
  - name: voxconverse
```

### 3. Dataset with Custom Parameters

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  feature_type: fbank
  num_mel_bins: 80

datasets:
  - name: ego4d
    download_params:
      dataset_parts: ["av_", "annotations"]
      install_cli: true
      timeout: 7200
    process_params:
      extract_audio: true
      audio_sample_rate: 16000
      min_segment_duration: 0.5
```

### 4. Different Feature Types

**Mel Filterbanks (Default)**:
```yaml
global_config:
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000
  frame_length: 0.025
  frame_shift: 0.01
```

**MFCCs**:
```yaml
global_config:
  feature_type: mfcc
  num_mel_bins: 40
  num_ceps: 13
  cepstral_lifter: 22
  use_energy: true
  sampling_rate: 16000
```

**Spectrograms**:
```yaml
global_config:
  feature_type: spectrogram
  num_filters: 257
  sampling_rate: 16000
  frame_length: 0.025
  frame_shift: 0.01
```

---

## Production Configurations

### 1. Multi-GPU Training

```yaml
training:
  epochs: 100
  batch_size: 64                # Per GPU
  
  optimizer:
    type: adamw
    lr: 0.0004                  # Scaled for 4 GPUs
    weight_decay: 0.01
  
  scheduler:
    type: cosine
    min_lr: 0.00001
    warmup_steps: 5000
  
  gradient_clipping: 1.0
  mixed_precision: true
  
  distributed:
    backend: nccl
    world_size: 4
    sync_gradient_barrier: true
    gradient_as_bucket_view: true
  
  checkpoint:
    save_dir: ./checkpoints/multi_gpu
    interval: 2000
    save_total_limit: 5
  
  logging:
    interval: 100
    tensorboard: true
    wandb: true
  
  performance:
    num_workers: 8              # Per GPU
    pin_memory: true
    persistent_workers: true
```

**Usage**:
```bash
torchrun --nproc_per_node=4 train.py --config configs/multi_gpu.yml
```

### 2. Mixed Precision Training

```yaml
training:
  mixed_precision: true
  amp_loss_scale: null          # Dynamic loss scaling
  
  gradient_clipping: 1.0        # Important for stability
  
  optimizer:
    type: adamw
    lr: 0.0001
    epsilon: 0.00000001         # Slightly larger epsilon for FP16
  
  checkpoint:
    save_dir: ./checkpoints/mixed_precision
    interval: 1000
```

### 3. Long Audio Sequences

For speaker diarization with long recordings:

```yaml
model:
  model_type: encoder
  encoder:
    d_model: 512
    num_layers: 12
    num_heads: 8
    attention_type: linear      # O(n) vs O(n²) for long sequences
    nb_features: 256

global_config:
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000
  frame_shift: 0.01             # 10ms frames = 100 frames/sec

training:
  batch_size: 8                 # Smaller batch for long sequences
  
  gradient_accumulation_steps: 8  # Maintain effective batch size
  
  eval_knobs:
    label_type: binary
    max_duration: 60.0          # Up to 60 seconds per batch
  
  performance:
    num_workers: 4
    pin_memory: true
```

---

## Troubleshooting Configs

### 1. Out of Memory

```yaml
training:
  batch_size: 8                 # ✅ Reduce batch size
  gradient_accumulation_steps: 8  # ✅ Maintain effective batch size
  mixed_precision: true         # ✅ Use FP16
  
  performance:
    num_workers: 2              # ✅ Reduce workers
  
  checkpoint:
    save_total_limit: 2         # ✅ Keep fewer checkpoints
```

### 2. Slow Training

```yaml
training:
  mixed_precision: true         # ✅ 2x speedup
  
  performance:
    num_workers: 8              # ✅ More data loading workers
    pin_memory: true            # ✅ Faster GPU transfer
    prefetch_factor: 4          # ✅ Prefetch more batches
    persistent_workers: true    # ✅ Keep workers alive
    compile_model: true         # ✅ torch.compile (PyTorch 2.0+)
```

### 3. Unstable Training (NaN Loss)

```yaml
training:
  gradient_clipping: 1.0        # ✅ Clip gradients
  
  optimizer:
    lr: 0.0001                  # ✅ Lower learning rate
    epsilon: 0.00000001         # ✅ Larger epsilon
  
  scheduler:
    warmup_steps: 1000          # ✅ Gradual warmup
  
  mixed_precision: false        # ✅ Disable FP16 if unstable
```

### 4. Overfitting

```yaml
model:
  global_config:
    dropout: 0.3                # ✅ Increase dropout

training:
  optimizer:
    weight_decay: 0.1           # ✅ Stronger regularization
  
  loss:
    label_smoothing: 0.1        # ✅ Label smoothing
    auxiliary:
      norm_reg: 0.1             # ✅ Norm regularization
  
  early_stopping:
    patience: 5                 # ✅ Stop when val loss plateaus
    metric: val_loss
```

### 5. Slow Convergence

```yaml
training:
  optimizer:
    lr: 0.001                   # ✅ Higher learning rate
    betas: [0.9, 0.98]          # ✅ Faster momentum
  
  scheduler:
    type: cosine
    warmup_steps: 2000          # ✅ Longer warmup
  
  batch_size: 64                # ✅ Larger batch
  gradient_accumulation_steps: 4
```

---

## Summary

### Configuration Templates

| Use Case | Config File | Key Features |
|----------|-------------|--------------|
| **Quick Test** | `minimal.yml` | Small model, CPU, fast |
| **Development** | `quick_experiment.yml` | Iterative, limited steps |
| **Diarization** | `encoder_diarization.yml` | Encoder, linear attention, BCE loss |
| **Language Model** | `decoder_lm.yml` | Causal decoder, cross-entropy |
| **Translation** | `encoder_decoder_translation.yml` | Full seq2seq |
| **Production** | `production.yml` | Multi-GPU, mixed precision, WandB |

### Next Steps

1. **Copy a template**: Start with the closest example
2. **Modify for your task**: Change model size, dataset, etc.
3. **Quick test**: Run for a few steps to validate
4. **Full training**: Scale up batch size, epochs, GPUs

### Further Reading

- [Configuration Concepts](./configuration-concepts.md) - Understand the system design
- [Comprehensive Reference](../../configs/comprehensive_all_options.yml) - All available options
- [Training README](./README.md) - Training pipeline overview

