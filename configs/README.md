# Configuration Files

This directory contains example configuration files for the TS-DIA training pipeline.

## 📚 Documentation

Before using these configs, read the documentation:

- **[Configuration Concepts](../docs/training/configuration-concepts.md)** - Understand the configuration system
- **[Configuration Examples](../docs/training/configuration-examples.md)** - Practical examples and use cases
- **[Training Pipeline](../docs/training/README.md)** - Training pipeline overview

## 📁 Configuration Files

### Reference Configuration

#### `comprehensive_all_options.yml` ⭐
**Complete reference with every available option documented**

This is the master reference showing ALL possible configuration options across:
- Model architecture (encoder, decoder, encoder-decoder)
- Dataset and feature extraction (25+ parameters)
- Training pipeline (optimizer, scheduler, loss, logging, etc.)

**Use this**: As a reference when creating new configurations. Copy sections you need.

---

### Model Architecture Examples

#### `encoder_model.yml`
**Encoder-only model for classification/diarization**

- Architecture: Encoder only
- Use cases: Speaker diarization, audio classification, sequence labeling
- Attention: Softmax (standard attention)
- Features: Simple, focused configuration

```bash
# Example usage
python train.py --config configs/encoder_model.yml
```

#### `decoder_model.yml`
**Decoder-only model for language modeling**

- Architecture: Decoder only
- Use cases: Language modeling, autoregressive generation
- Attention: Causal linear (efficient for autoregressive tasks)
- Features: No cross-attention

```bash
# Example usage
python train.py --config configs/decoder_model.yml
```

#### `encoder_decoder_model.yml`
**Encoder-decoder model for sequence-to-sequence tasks**

- Architecture: Encoder-decoder
- Use cases: Translation, summarization, speech recognition
- Attention: Linear (encoder) + Causal linear (decoder)
- Features: Cross-attention enabled, comprehensive model configuration

```bash
# Example usage
python train.py --config configs/encoder_decoder_model.yml
```

---

### Complete Experiment Configurations

#### `training_example.yml` ⭐
**Comprehensive training configuration with all features**

Complete configuration demonstrating:
- Full model specification
- Dataset with feature extraction
- Comprehensive training setup (optimizer, scheduler, mixed precision)
- Validation and early stopping
- Checkpointing and logging (TensorBoard, WandB)
- Distributed training settings

**Use this**: As a starting point for production experiments.

```bash
# Full training run
python train.py --config configs/training_example.yml

# Multi-GPU training
accelerate launch train.py --config configs/training_example.yml
```

#### `training_simple.yml`
**Minimal essential training configuration**

Simplified configuration with only essential settings:
- Basic optimizer and scheduler
- Simple checkpointing
- Minimal logging
- Good for learning and quick experiments

**Use this**: When you want a clean, simple starting point.

```bash
# Simple training
python train.py --config configs/training_simple.yml
```

#### `training_minimal.yml`
**Absolute minimum for quick testing**

Ultra-minimal configuration for:
- Fast iteration during development
- Quick validation of code changes
- Testing new features
- CPU-based testing

Features:
- Small model (d_model=64, 2 layers)
- Tiny dataset (yesno)
- max_steps=3 (stops after 3 steps)
- CPU device

**Use this**: For quick smoke tests and development.

```bash
# Quick test
python train.py --config configs/training_minimal.yml
```

#### `full_experiment.yml`
**Complete experiment: model + dataset + training**

Demonstrates a complete experimental setup:
- Model configuration (encoder-decoder)
- Multiple datasets
- Training configuration

**Use this**: As an example of a full experiment specification.

---

## 🚀 Quick Start

### 1. Choose a Configuration

Pick based on your use case:

| Use Case | Config File | Description |
|----------|-------------|-------------|
| **Quick Test** | `training_minimal.yml` | Fast validation, CPU, 3 steps |
| **Learning** | `training_simple.yml` | Simple but complete |
| **Production** | `training_example.yml` | Full-featured setup |
| **Encoder Only** | `encoder_model.yml` | Diarization, classification |
| **Decoder Only** | `decoder_model.yml` | Language modeling |
| **Seq2Seq** | `encoder_decoder_model.yml` | Translation, summarization |
| **Complete Experiment** | `full_experiment.yml` | Full setup example |
| **Reference** | `comprehensive_all_options.yml` | All options documented |

### 2. Customize

Edit the config to match your needs:

```yaml
# Change model size
model:
  encoder:
    d_model: 256        # Your choice
    num_layers: 8       # Your choice

# Change dataset
datasets:
  - name: ami           # Your dataset

# Adjust training
training:
  batch_size: 32       # Based on your GPU
  epochs: 100          # Your target
```

### 3. Run Training

```bash
# Basic training
python train.py --config configs/your_config.yml

# With CLI overrides
python train.py --config configs/your_config.yml \
  --batch-size 64 \
  --lr 0.0001 \
  --epochs 100

# Multi-GPU with Accelerate
accelerate launch train.py --config configs/your_config.yml

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 train.py --config configs/your_config.yml
```

## 📖 Configuration Structure

All configuration files follow this structure:

```yaml
# 1. Model Configuration
model:
  model_type: encoder | decoder | encoder_decoder
  name: my_model
  
  global_config:    # Shared parameters
    dropout: 0.1
    batch_size: 32
    d_ff: 4
    device: cpu
  
  encoder:          # Encoder-specific
    # ...
  
  decoder:          # Decoder-specific (if applicable)
    # ...

# 2. Global Dataset Configuration
global_config:
  corpus_dir: ./data
  output_dir: ./manifests
  
  # Feature extraction (unified)
  feature_type: fbank
  num_mel_bins: 80
  sampling_rate: 16000
  # ... more feature options

# 3. Dataset List
datasets:
  - name: dataset_name
    download_params:  # Optional
      # ...
    process_params:   # Optional
      # ...

# 4. Training Configuration
training:
  epochs: 100
  batch_size: 32
  
  optimizer:
    type: adamw
    lr: 0.0001
    # ...
  
  scheduler:
    type: cosine
    # ...
  
  # ... more training options
```

## 🎯 Best Practices

### 1. Start with Examples

Don't write configs from scratch:

```bash
# Copy an example
cp configs/training_simple.yml configs/my_experiment.yml

# Edit your copy
vim configs/my_experiment.yml
```

### 2. Test First

Before long runs, test with minimal config:

```bash
# Quick validation (3 steps)
python train.py --config configs/my_experiment.yml --max-steps 3
```

### 3. Use Defaults

Only specify what you need to change:

```yaml
# ❌ Don't do this (too verbose)
optimizer:
  type: adamw
  lr: 0.0001
  weight_decay: 0.01
  betas: [0.9, 0.999]
  epsilon: 0.00000001
  amsgrad: false
  momentum: null
  nesterov: false

# ✅ Do this (only what changes)
optimizer:
  type: adamw
  lr: 0.0001
  weight_decay: 0.01
```

### 4. Document Your Choices

Add comments explaining your settings:

```yaml
training:
  batch_size: 64        # Reduced from 128 due to GPU memory
  
  optimizer:
    lr: 0.0002          # Found via grid search
    weight_decay: 0.01  # Helps prevent overfitting on small dataset
```

### 5. Version Control

Track your configs:

```bash
git add configs/my_experiment.yml
git commit -m "Add config for speaker diarization experiment"
```

## 🔧 Common Modifications

### Change Model Size

```yaml
model:
  encoder:
    d_model: 768        # Increase from 512
    num_layers: 12      # Increase from 6
```

### Use Different Dataset

```yaml
datasets:
  - name: ami          # Change to your dataset
```

### Adjust Training

```yaml
training:
  batch_size: 64       # Based on GPU memory
  epochs: 100          # Based on dataset size
  
  optimizer:
    lr: 0.0001         # Tune for your task
```

### Enable Multi-GPU

```yaml
training:
  distributed:
    backend: nccl
    world_size: 4      # Number of GPUs
```

### Add WandB Logging

```yaml
training:
  logging:
    wandb: true
    wandb_project: my-project
    wandb_entity: my-team
```

## 🐛 Troubleshooting

### Out of Memory

```yaml
training:
  batch_size: 8                    # Reduce batch size
  gradient_accumulation_steps: 8   # Maintain effective batch size
  mixed_precision: true            # Use FP16
```

### Training Too Slow

```yaml
training:
  mixed_precision: true
  
  performance:
    num_workers: 8                 # More data workers
    compile_model: true            # PyTorch 2.0+ compilation
```

### NaN Loss

```yaml
training:
  gradient_clipping: 1.0           # Clip gradients
  
  optimizer:
    lr: 0.00001                    # Lower learning rate
  
  scheduler:
    warmup_steps: 1000             # Gradual warmup
```

## 📚 Additional Resources

- **[Configuration Concepts](../docs/training/configuration-concepts.md)** - System design
- **[Configuration Examples](../docs/training/configuration-examples.md)** - Practical examples
- **[Training Pipeline](../docs/training/README.md)** - Pipeline overview
- **[Model Factory Guide](../docs/overview/MODEL_FACTORY_GUIDE.md)** - Model creation

## 💡 Need Help?

1. Check **[Configuration Examples](../docs/training/configuration-examples.md)** for your use case
2. Review **[comprehensive_all_options.yml](./comprehensive_all_options.yml)** for all available options
3. Look at similar configs in this directory
4. Read the **[Configuration Concepts](../docs/training/configuration-concepts.md)** guide

