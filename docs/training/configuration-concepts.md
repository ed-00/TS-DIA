# Configuration System - Core Concepts

This document explains the design philosophy and core concepts of the TS-DIA configuration system.

## Table of Contents

1. [Overview](#overview)
2. [Configuration Architecture](#configuration-architecture)
3. [Three-Part Configuration](#three-part-configuration)
4. [Type Safety & Validation](#type-safety--validation)
5. [Global Configuration Pattern](#global-configuration-pattern)
6. [Configuration Inheritance](#configuration-inheritance)
7. [Best Practices](#best-practices)

---

## Overview

The TS-DIA configuration system is built on three core principles:

1. **Single Source of Truth**: One YAML file defines everything (model, data, training)
2. **Type Safety**: All configs map to typed dataclasses with validation
3. **Sensible Defaults**: Only specify what you need to change

### Design Goals

- **Simplicity**: Easy to read and write configurations
- **Flexibility**: Support simple to complex use cases
- **Reproducibility**: Configurations fully capture experimental setup
- **Extensibility**: Easy to add new options without breaking existing configs

---

## Configuration Architecture

```
┌─────────────────────────────────────────────────┐
│          config.yml (User Input)                │
│  - Human-readable YAML                          │
│  - Partial configuration (only what changes)    │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│        unified_parser() (parse_args.py)         │
│  - Parses YAML                                  │
│  - Merges with defaults                         │
│  - Validates structure                          │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Typed Dataclasses (Validation)          │
│  - ModelConfig (model/model_types.py)           │
│  - DatasetConfig (data_manager/dataset_types.py)│
│  - TrainingConfig (training/config.py)          │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│          Runtime Components                     │
│  - Model Factory                                │
│  - Data Manager                                 │
│  - Trainer                                      │
└─────────────────────────────────────────────────┘
```

### Key Components

1. **YAML Parser** (`parse_args.py`):
   - Reads user configuration
   - Merges with defaults
   - Creates typed dataclass instances

2. **Configuration Dataclasses**:
   - `ModelConfig`: Model architecture parameters
   - `DatasetConfig`: Dataset and feature extraction settings
   - `TrainingConfig`: Training loop, optimization, logging

3. **Factory Pattern**:
   - `create_model()`: Builds model from ModelConfig
   - `DatasetManager.load_datasets()`: Loads data from DatasetConfig
   - `Trainer()`: Runs training from TrainingConfig

---

## Three-Part Configuration

Every configuration file has three main sections:

### 1. Model Configuration

Defines the neural network architecture.

```yaml
model:
  model_type: encoder_decoder  # Architecture type
  name: my_model               # Model name
  
  global_config:               # Shared parameters
    dropout: 0.1
    batch_size: 32
    d_ff: 4
    device: cpu
  
  encoder:                     # Encoder-specific
    d_model: 512
    num_layers: 6
    num_heads: 8
    attention_type: linear
    activation: GEGLU
  
  decoder:                     # Decoder-specific
    d_model: 512
    num_layers: 6
    num_heads: 8
    attention_type: causal_linear
    activation: SWIGLU
```

**Maps to**: `ModelConfig` dataclass in `model/model_types.py`

**Key Concepts**:
- **model_type**: Determines architecture (`encoder`, `decoder`, `encoder_decoder`)
- **global_config**: Shared parameters across encoder/decoder
- **Component configs**: Specific to each model component

### 2. Dataset Configuration

Defines data sources and feature extraction.

```yaml
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  
  # Feature extraction (unified across all datasets)
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000
  frame_length: 0.025
  frame_shift: 0.01

datasets:
  - name: ami
  - name: librispeech
    download_params:
      dataset_parts: mini_librispeech
```

**Maps to**: `GlobalConfig` and `DatasetConfig` in `data_manager/dataset_types.py`

**Key Concepts**:
- **global_config**: Default settings for all datasets
- **Feature configuration**: Audio processing parameters
- **Per-dataset overrides**: Dataset-specific parameters

### 3. Training Configuration

Defines the training loop and optimization.

```yaml
training:
  epochs: 100
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
    main: cross_entropy
    label_smoothing: 0.1
  
  checkpoint:
    save_dir: ./checkpoints
    interval: 1000
  
  logging:
    interval: 50
    tensorboard: true
```

**Maps to**: `TrainingConfig` in `training/config.py`

**Key Concepts**:
- **Nested configuration**: Each aspect (optimizer, scheduler, etc.) is its own config
- **Optional sections**: Only include what you need
- **Comprehensive control**: From basic to advanced features

---

## Type Safety & Validation

### Dataclass-Based Configuration

All configurations are typed Python dataclasses:

```python
@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str                                    # Required
    lr: float                                    # Required
    weight_decay: Optional[float] = 0.0          # Optional with default
    betas: Optional[Tuple[float, float]] = (0.9, 0.999)
    momentum: Optional[float] = None
```

**Benefits**:
- **Type checking**: Catch errors before runtime
- **Auto-completion**: IDEs can suggest fields
- **Documentation**: Self-documenting with type hints
- **Validation**: Automatic type conversion and validation

### Validation Layers

1. **YAML Parsing**: Basic syntax validation
2. **Dataclass Creation**: Type validation via `dacite`
3. **Custom Validation**: Model-specific validation methods

Example validation:

```python
@dataclass
class ModelConfig:
    model_type: ModelType
    config: Union[EncoderConfig, DecoderConfig, EncoderDecoderConfig]
    
    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.model_type == "encoder" and not isinstance(self.config, EncoderConfig):
            raise ValueError(f"model_type 'encoder' requires EncoderConfig")
```

### Error Messages

The system provides clear error messages:

```python
# Missing required field
ValueError: Missing required field 'lr' in optimizer config

# Type mismatch
ValueError: Expected int for 'num_layers', got str: 'six'

# Invalid value
ValueError: attention_type must be one of: softmax, linear, causal_linear
```

---

## Global Configuration Pattern

The **Global Config Pattern** provides defaults that cascade to all datasets.

### How It Works

1. Define global defaults in `global_config`
2. Each dataset inherits these defaults
3. Datasets can override specific values

```yaml
global_config:
  corpus_dir: ./data           # ← Global default
  output_dir: ./manifests      # ← Global default
  
  feature_type: fbank          # ← Applied to all datasets
  num_mel_bins: 80
  sampling_rate: 16000

datasets:
  - name: ami                  # Uses all global defaults
  
  - name: timit
    process_params:
      num_phones: 48           # Override specific parameter
      # Still inherits corpus_dir, output_dir, features
```

### Feature Configuration Unification

All datasets share the same feature extraction parameters:

```yaml
global_config:
  # Basic parameters
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000
  frame_length: 0.025
  frame_shift: 0.01
  
  # Advanced parameters (optional)
  preemph_coeff: 0.97
  window_type: povey
  dither: 0.0
```

**Benefits**:
- **Consistency**: All datasets processed identically
- **Reproducibility**: Single source of truth for features
- **Simplicity**: Don't repeat feature config per dataset

### Implementation

The `GlobalConfig` dataclass wraps a `FeatureConfig`:

```python
@dataclass
class GlobalConfig:
    corpus_dir: str = "./data"
    output_dir: str = "./manifests"
    force_download: bool = False
    features: FeatureConfig = field(default_factory=FeatureConfig)
    
    def get_feature_config(self) -> FeatureConfig:
        """Get feature extraction configuration"""
        return self.features
```

Datasets apply global config:

```python
dataset_config.apply_global_config(global_config)
# → Merges global settings into dataset-specific config
```

---

## Configuration Inheritance

### Model Global Config

Shared parameters between encoder and decoder:

```yaml
model:
  global_config:
    dropout: 0.1      # Applied to both encoder and decoder
    batch_size: 32
    d_ff: 4
    device: cpu
  
  encoder:
    d_model: 512      # Encoder-specific
    # dropout, d_ff inherited from global_config
  
  decoder:
    d_model: 512      # Decoder-specific
    # dropout, d_ff inherited from global_config
```

### Dataset Config Inheritance

```yaml
global_config:
  corpus_dir: ./data
  feature_type: fbank
  num_mel_bins: 80

datasets:
  - name: ami
    # Inherits: corpus_dir, feature_type, num_mel_bins
  
  - name: librispeech
    download_params:
      dataset_parts: mini_librispeech
    # Also inherits: corpus_dir, feature_type, num_mel_bins
    # Adds: dataset_parts override
```

### Precedence Rules

1. **Most Specific Wins**: Component-specific > Global > Default
2. **Explicit Overrides**: User values > Default values
3. **Type Requirements**: Required fields must be provided

Example:

```yaml
# Default: dropout = 0.0
# Global: dropout = 0.1
# Component: dropout = 0.2

model:
  global_config:
    dropout: 0.1          # ← Overrides default (0.0)
  
  encoder:
    dropout: 0.2          # ← Overrides global (0.1)
    # Result: encoder uses 0.2
  
  decoder:
    # No explicit dropout
    # Result: decoder uses global 0.1
```

---

## Best Practices

### 1. Start Simple, Add Complexity

Begin with minimal config:

```yaml
# Minimal: Only essentials
training:
  epochs: 10
  batch_size: 32
  
  optimizer:
    type: adamw
    lr: 0.0001
  
  scheduler:
    type: cosine
```

Add complexity as needed:

```yaml
# Complex: Full control
training:
  epochs: 100
  batch_size: 32
  gradient_clipping: 1.0
  mixed_precision: true
  
  optimizer:
    type: adamw
    lr: 0.0001
    weight_decay: 0.01
    betas: [0.9, 0.999]
  
  scheduler:
    type: cosine
    min_lr: 0.000001
    warmup_steps: 1000
  
  early_stopping:
    patience: 5
    metric: val_loss
```

### 2. Use Comments

Document your choices:

```yaml
training:
  batch_size: 64        # Reduced from 128 due to memory constraints
  
  optimizer:
    lr: 0.0002          # Found via grid search
    weight_decay: 0.01  # Helps with overfitting
```

### 3. Reference Configurations

Keep reference configs for comparison:

```
configs/
├── comprehensive_all_options.yml   # Full reference
├── training_simple.yml             # Minimal example
├── training_example.yml            # Typical setup
└── my_experiment.yml               # Your experiment
```

### 4. Version Control

Track configs with git:

```bash
git add configs/my_experiment.yml
git commit -m "Add config for experiment: BERT-style pretraining"
```

### 5. Naming Conventions

```
configs/
├── {model_type}_model.yml          # Model architecture examples
├── {dataset}_example.yml           # Dataset-specific examples
├── {experiment_name}.yml           # Your experiments
└── comprehensive_all_options.yml   # Complete reference
```

### 6. Separation of Concerns

Keep different aspects separate:

```yaml
# Model architecture
model:
  model_type: encoder
  encoder:
    d_model: 512
    num_layers: 6

# Data and features
global_config:
  feature_type: fbank
  num_mel_bins: 80

# Training hyperparameters
training:
  optimizer:
    lr: 0.0001
```

### 7. Use Defaults Wisely

Only specify what you need to change:

```yaml
# ❌ Too verbose
scheduler:
  type: cosine
  min_lr: 0.0
  max_lr: null
  warmup_steps: 0
  decay_steps: null
  num_cycles: 1
  gamma: 0.1

# ✅ Better: Only specify what changes
scheduler:
  type: cosine
  min_lr: 0.000001
  warmup_steps: 1000
```

### 8. Validate Early

Test configs before long training runs:

```bash
# Quick validation
python train.py --config configs/my_experiment.yml --max-steps 10
```

### 9. Document Experiments

Include metadata in configs:

```yaml
# Experiment: BERT-style pretraining on AMI
# Date: 2025-10-14
# Hypothesis: Linear attention can match softmax performance
# Previous: configs/experiment_001.yml

model:
  model_type: encoder
  # ...
```

---

## Advanced Topics

### CLI Overrides

Override config values from command line:

```bash
python train.py --config configs/base.yml \
  --lr 0.0001 \
  --batch-size 64 \
  --epochs 100
```

### Config Composition

Reference configs can be composed:

```python
# Load base config
base_config = load_config("configs/base.yml")

# Override specific values
base_config.training.optimizer.lr = 0.0001
base_config.training.batch_size = 64

# Use modified config
trainer = Trainer(model, train_dl, val_dl, config=base_config)
```

### Hyperparameter Tuning

Integrate with Ray Tune or Optuna:

```yaml
training:
  tuning:
    library: raytune
    search_space:
      lr: [0.00001, 0.0001, 0.0005]
      batch_size: [32, 64]
      dropout: [0.1, 0.2, 0.3]
```

---

## Summary

The TS-DIA configuration system provides:

✅ **Single file** for complete experimental setup  
✅ **Type safety** via dataclasses  
✅ **Sensible defaults** - only specify what changes  
✅ **Global configs** for consistency  
✅ **Validation** to catch errors early  
✅ **Extensibility** for new features  

**Next Steps**:
- See [Configuration Examples](./configuration-examples.md) for practical examples
- See [Configuration Reference](./comprehensive_all_options.yml) for all options
- See [Training README](./README.md) for training pipeline overview

