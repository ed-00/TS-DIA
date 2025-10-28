# Universal Training Pipeline

This document describes the comprehensive training pipeline for deep learning models with full support for distributed training, mixed precision, and diarization tasks.

## 📚 Documentation Index

- **[Configuration Concepts](./configuration-concepts.md)** - Understand the configuration system design
- **[Configuration Examples](./configuration-examples.md)** - Practical examples for common use cases
- **[Comprehensive Reference](../../configs/comprehensive_all_options.yml)** - All available configuration options
- **This Document** - Training pipeline overview and API reference

## Overview

The training pipeline provides:
- **Unified Configuration**: Single YAML file for model, dataset, and training settings
- **HuggingFace Accelerate Integration**: Distributed training, mixed precision, gradient accumulation
- **Diarization Support**: Custom dataloaders for speaker diarization using Lhotse CutSets
- **Comprehensive Logging**: TensorBoard and WandB integration via Accelerate
- **Advanced Features**: Early stopping, callbacks, feature redrawing for Performer models
- **Reproducibility**: Seed management, checkpoint saving/loading

## Quick Start

### Basic Training

```bash
# Train with configuration file
python train.py --config configs/training_example.yml

# Distributed training with Accelerate
accelerate launch train.py --config configs/training_example.yml

# Multi-GPU training
torchrun --nproc_per_node=4 train.py --config configs/training_example.yml
```

## Configuration

> 📖 **Detailed Documentation**: See [Configuration Concepts](./configuration-concepts.md) for system design and [Configuration Examples](./configuration-examples.md) for practical examples.

All configuration comes from YAML files with three main sections:

### 1. Model Configuration

```yaml
model:
  model_type: encoder_decoder
  name: performer_translator
  
  global_config:
    dropout: 0.1
    batch_size: 32
    d_ff: 4
  
  encoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    attention_type: linear
  
  decoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    attention_type: causal_linear
```

### 2. Dataset Configuration

The `global_config` section unifies feature extraction across all datasets:

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  
  # Feature extraction configuration (unified for all datasets)
  feature_type: fbank  # Options: fbank, mfcc, spectrogram
  
  # Basic parameters
  num_mel_bins: 80
  frame_length: 0.025  # 25ms
  frame_shift: 0.01  # 10ms
  sampling_rate: 16000
  
  # Window and preprocessing (optional)
  preemph_coeff: 0.97  # Pre-emphasis
  window_type: povey  # povey, hanning, hamming
  dither: 0.0
  snip_edges: false
  remove_dc_offset: true
  
  # Mel filterbank (optional)
  low_freq: 20.0
  high_freq: -400.0  # -400 = nyquist - 400
  
  # MFCC-specific (when feature_type: mfcc)
  num_ceps: 13
  cepstral_lifter: 22
  use_energy: false
  
  # Feature computation
  storage_path: null  # null = in-memory
  num_jobs: 1  # Parallel jobs
  progress_bar: true

datasets:
  - name: ami
```

**Feature Types:**
- `fbank`: Mel-frequency filter banks (80 bins default)
- `mfcc`: Mel-frequency cepstral coefficients (13 coefficients default)
- `spectrogram`: Power spectrogram

**Key Parameters:**
- **Basic**: `num_mel_bins`, `frame_length`, `frame_shift`, `sampling_rate`
- **Window**: `window_type` (povey/hanning/hamming), `preemph_coeff`, `dither`
- **Mel filters**: `low_freq`, `high_freq`, `num_filters`
- **MFCC**: `num_ceps`, `cepstral_lifter`, `use_energy`
- **Storage**: `storage_path` (null=memory), `num_jobs`, `storage_type`

All parameters have sensible defaults - only specify what you need to change!

### 3. Training Configuration

```yaml
training:
  # Basic settings
  epochs: 50
  batch_size: 32
  random_seed: 42
  
  # Optimizer
  optimizer:
    type: adamw
    lr: 2e-4
    weight_decay: 0.01
    betas: [0.9, 0.999]
  
  # Scheduler
  scheduler:
    type: cosine
    min_lr: 1e-6
    warmup_steps: 1000
  
  # Training behavior
  gradient_clipping: 1.0
  gradient_accumulation_steps: 4
  mixed_precision: true
  
  # Performer-specific
  feature_redraw_interval: 100
  fixed_projection: false
  
  # Loss configuration
  loss:
    main: cross_entropy
    label_smoothing: 0.1
    auxiliary:
      norm_reg: 0.1
  
  # Validation
  validation:
    interval: 500
    batch_size: 64
  
  # Early stopping
  early_stopping:
    patience: 5
    metric: val_loss
    min_delta: 0.001
    mode: min
  
  # Checkpointing
  checkpoint:
    save_dir: ./checkpoints
    interval: 1000
    save_total_limit: 5
    resume: null
  
  # Logging (via Accelerate)
  logging:
    interval: 50
    tensorboard: true
    wandb: true
    wandb_project: my-project
    log_metrics: [loss, accuracy]
  
  # Performance
  performance:
    num_workers: 4
    pin_memory: true
  
  # Diarization settings
  eval_knobs:
    label_type: binary  # binary, speaker_id, or custom
    max_duration: null  # for dynamic bucketing
```

## Diarization DataLoaders

The pipeline includes specialized dataloaders for speaker diarization tasks using Lhotse:

### Label Types

1. **Binary**: Binary labels (1 if any speaker active, 0 otherwise)
2. **Speaker ID**: Multi-class labels with speaker IDs per frame
3. **Custom**: Extensible format for custom label configurations

### Configuration

```yaml
training:
  eval_knobs:
    label_type: binary
    max_duration: 30.0  # max seconds per batch
```

### DataLoader Features

- Variable-length sequence handling
- Dynamic bucketing for efficient batching
- Frame-level diarization labels
- Attention mask generation
- Lhotse CutSet integration

## Logging with Accelerate

All logging goes through HuggingFace Accelerate for proper distributed training support:

### TensorBoard

```yaml
logging:
  tensorboard: true
  interval: 50
```

### Weights & Biases

```yaml
logging:
  wandb: true
  wandb_project: my-diarization-project
  wandb_entity: my-team
```

### Logged Metrics

- Training loss (main + auxiliary)
- Validation metrics
- Learning rate
- Gradient norms (optional)
- System metrics

## Advanced Features

### Callbacks

The pipeline supports extensible callbacks:

- **EarlyStoppingCallback**: Stop training when metric stops improving
- **FeatureRedrawCallback**: Redraw Performer projection matrices
- **LRMonitorCallback**: Monitor learning rate changes
- **GradientClippingCallback**: Apply gradient clipping

### Checkpointing

Automatic checkpoint management with:
- Regular interval saving
- Best model tracking
- Optimizer/scheduler state
- Performer feature matrices
- Resume training support

### Mixed Precision

FP16 training via Accelerate:

```yaml
training:
  mixed_precision: true
  amp_loss_scale: null  # dynamic scaling
```

### Distributed Training

```yaml
training:
  distributed:
    backend: nccl
    world_size: 8
    sync_gradient_barrier: true
```

## Architecture

### Core Components

1. **Trainer** (`training/trainer.py`): Main training loop
2. **Config** (`training/config.py`): Type-safe configuration dataclasses
3. **Accelerate Utils** (`training/accelerate_utils.py`): Distributed training utilities
4. **Diarization DataLoader** (`training/diarization_dataloader.py`): Lhotse-based dataloaders
5. **Logging** (`training/logging_utils.py`): Accelerate-integrated logging
6. **Callbacks** (`training/callbacks.py`): Extensible callback system
7. **Losses** (`training/losses.py`): Loss functions and metrics
8. **Optimizers** (`training/optimizers.py`): Optimizer and scheduler factories

### Parser Integration

The `unified_parser()` in `parse_args.py` handles all configuration:

```python
from parse_args import unified_parser

args, model_config, dataset_configs, training_config = unified_parser()
```

## Examples

### Minimal Configuration

```yaml
training:
  epochs: 10
  batch_size: 32
  
  optimizer:
    type: adamw
    lr: 1e-4
  
  scheduler:
    type: cosine
    min_lr: 1e-6
  
  checkpoint:
    save_dir: ./checkpoints
    interval: 1000
```

### Full Configuration

See the following for comprehensive examples:
- **[Configuration Examples](./configuration-examples.md)** - Practical configurations for different use cases
- **[configs/comprehensive_all_options.yml](../../configs/comprehensive_all_options.yml)** - Complete reference with all options
- **[configs/training_example.yml](../../configs/training_example.yml)** - Typical full-featured setup
- **[configs/training_simple.yml](../../configs/training_simple.yml)** - Minimal essential configuration

## Extending the Pipeline

### Custom Loss Functions

```python
from training.losses import LossRegistry

class MyCustomLoss(nn.Module):
    def forward(self, outputs, targets):
        # ... custom logic
        return loss

LossRegistry.register("my_loss", MyCustomLoss)
```

### Custom Callbacks

```python
from training.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_end(self, trainer, epoch, metrics):
        # ... custom logic
        pass
```

### Custom Label Types

Extend `DiarizationCollator` for custom diarization labels:

```python
class CustomCollator(DiarizationCollator):
    def __call__(self, cuts):
        # ... custom label generation
        return batch
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or increase `gradient_accumulation_steps`
2. **Slow DataLoader**: Increase `num_workers` in performance config
3. **NaN Loss**: Reduce learning rate or enable gradient clipping
4. **Distributed Hangs**: Check `sync_gradient_barrier` setting

### Debug Mode

Enable profiling for performance analysis:

```yaml
training:
  profiling: true
```

## Dependencies

Required packages:
- `torch >= 2.0.0`
- `accelerate >= 0.26.0`
- `lhotse >= 1.31.0`
- `tensorboard >= 2.14.0` (optional)
- `wandb >= 0.16.0` (optional)
- `dacite >= 1.8.0`

Install with:
```bash
pip install -r requirements.txt
```

## Additional Resources

### Configuration Documentation
- [Configuration Concepts](./configuration-concepts.md) - System design and best practices
- [Configuration Examples](./configuration-examples.md) - Real-world configuration examples
- [Comprehensive Reference](../../configs/comprehensive_all_options.yml) - All available options

### Example Configurations
Located in `configs/`:
- **`comprehensive_all_options.yml`** - Complete reference with every option documented
- **`training_simple.yml`** - Minimal configuration
- **`training_example.yml`** - Typical full-featured setup
- **`encoder_model.yml`** - Encoder-only architecture example
- **`decoder_model.yml`** - Decoder-only architecture example
- **`encoder_decoder_model.yml`** - Encoder-decoder architecture example
- **`full_experiment.yml`** - Complete experiment with model + data + training
- **`training_minimal.yml`** - Quick test configuration

### Model Documentation
- [Model Factory Guide](../overview/MODEL_FACTORY_GUIDE.md)
- [Encoder-Decoder Guide](../overview/ENCODER_DECODER_GUIDE.md)
- [Normalization Guide](../overview/normalization_guide.md)
- [Positional Encoding Guide](../overview/positional_encoding_guide.md)

