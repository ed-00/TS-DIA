# Model Factory Pattern Guide

## Overview

The model factory pattern provides a unified interface for creating transformer models from YAML configurations. It follows the same design philosophy as the data manager, with type-safe configurations and extensible architecture.

## Features

✅ **Type-Safe Model Creation** - All parameters validated at parse time  
✅ **YAML Configuration** - Define models in human-readable format  
✅ **Multiple Model Types** - Encoder, Decoder, Encoder-Decoder  
✅ **Combined Parsing** - Model + Dataset configs in one file  
✅ **Global Defaults** - Share common parameters across encoder/decoder  
✅ **Extensible Design** - Easy to add new model types  

## Quick Start

### 1. Create a Model Configuration

```yaml
# configs/my_model.yml
model:
  model_type: encoder_decoder
  name: translator
  
  global_config:
    dropout: 0.1
    batch_size: 32
  
  encoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    d_ff: 4
    attention_type: linear
    activation: GEGLU
  
  decoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    d_ff: 4
    attention_type: causal_linear
    activation: SWIGLU
```

### 2. Load and Create Model

```python
from model.model_factory import create_model
from model.parse_model_args import parse_model_config

# Parse configuration
config = parse_model_config('configs/my_model.yml')

# Create model
model = create_model(config)

# Use model
output = model(source, target)
```

## Model Types

### Encoder-Only

**Use for**: Text classification, feature extraction, embedding

```yaml
model:
  model_type: encoder
  name: classifier
  
  encoder:
    d_model: 256
    device: cpu
    batch_size: 32
    num_layers: 4
    num_heads: 8
    d_ff: 4
    dropout: 0.1
    attention_type: softmax
    activation: GELU
```

**Python Usage**:
```python
encoder = create_model(config)
encoded = encoder(input_embeddings)
```

### Decoder-Only

**Use for**: Language modeling, autoregressive generation

```yaml
model:
  model_type: decoder
  name: language_model
  
  decoder:
    d_model: 512
    device: cpu
    batch_size: 32
    num_layers: 12
    num_heads: 8
    d_ff: 4
    dropout: 0.1
    attention_type: causal_linear
    activation: SWIGLU
    nb_features: 256
    use_cross_attention: false  # No encoder
```

**Python Usage**:
```python
lm = create_model(config)
output = lm(tokens, use_causal_mask=True)
```

### Encoder-Decoder

**Use for**: Translation, summarization, seq2seq tasks

```yaml
model:
  model_type: encoder_decoder
  name: seq2seq
  
  encoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    # ... encoder params
  
  decoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    # ... decoder params
```

**Python Usage**:
```python
seq2seq = create_model(config)
output = seq2seq(source, target)
```

## Configuration Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Model dimension |
| `device` | str | Device (cpu, cuda, cuda:0) |
| `batch_size` | int | Batch size |
| `num_layers` | int | Number of transformer layers |
| `num_heads` | int | Number of attention heads |
| `d_ff` | int | FFN expansion factor |
| `dropout` | float | Dropout rate |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention_type` | str | "softmax" | softmax, linear, causal_linear |
| `activation` | str | "GELU" | GELU, RELU, GEGLU, SWIGLU, etc. |
| `nb_features` | int | None | Random features for linear attention |
| `use_cross_attention` | bool | False (decoder only) | Enable cross-attention |
| `use_rezero` | bool | False | Use ReZero normalization |
| `use_scalenorm` | bool | False | Use PreScaleNorm |
| `feature_redraw_interval` | int | 1000 | Projection redraw interval |
| `auto_check_redraw` | bool | True | Auto redraw projections |

## Global Configuration

Share common parameters across encoder and decoder:

```yaml
model:
  model_type: encoder_decoder
  
  global_config:
    dropout: 0.1
    batch_size: 32
    d_ff: 4
    device: cpu
  
  encoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    attention_type: linear
    # dropout, batch_size, d_ff, device inherited
  
  decoder:
    d_model: 512
    num_layers: 6
    num_heads: 8
    attention_type: causal_linear
    # dropout, batch_size, d_ff, device inherited
```

## Combined Model + Dataset Configuration

Put both model and dataset configs in one file:

```yaml
# configs/experiment.yml

# Model configuration
model:
  model_type: encoder_decoder
  name: translator
  
  encoder:
    d_model: 512
    # ... params
  
  decoder:
    d_model: 512
    # ... params

# Dataset configuration
global_config:
  corpus_dir: ./data
  output_dir: ./manifests

datasets:
  - name: librispeech
  - name: timit
```

**Parse both**:
```python
from parse_args import parse_config
from model.model_factory import create_model
from data_manager.data_manager import DatasetManager

# Parse combined config
model_config, dataset_configs = parse_config('configs/experiment.yml')

# Create model
model = create_model(model_config)

# Load datasets
cut_sets = DatasetManager.load_datasets(datasets=dataset_configs)
```

## Command Line Usage

### Model Only
```bash
python -m model.parse_model_args --config configs/my_model.yml
```

### Combined (Model + Datasets)
```bash
python parse_args.py --config configs/experiment.yml
```

### Combined with Filters
```bash
# Only parse model
python parse_args.py --config configs/experiment.yml --model-only

# Only parse datasets
python parse_args.py --config configs/experiment.yml --data-only
```

## Factory Pattern API

### ModelFactory Class

```python
from model.model_factory import ModelFactory

# Create encoder
encoder = ModelFactory.create_encoder(encoder_config)

# Create decoder
decoder = ModelFactory.create_decoder(decoder_config)

# Create encoder-decoder
seq2seq = ModelFactory.create_encoder_decoder(enc_dec_config)

# Create from ModelConfig (auto-detects type)
model = ModelFactory.create_model(model_config)

# List available types
types = ModelFactory.list_model_types()
# ['encoder', 'decoder', 'encoder_decoder']
```

### Configuration Classes

```python
from model.model_types import (
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    EncoderDecoderConfig
)

# Programmatic configuration
from model.types import PerformerParams
from model.activations import ActivationFunctions

encoder_params = PerformerParams(
    d_model=512,
    device=torch.device('cuda'),
    batch_size=32,
    num_layers=6,
    num_heads=8,
    d_ff=4,
    dropout=0.1,
    attention_type='linear',
    activation=ActivationFunctions.GEGLU
)

config = ModelConfig(
    model_type='encoder',
    config=EncoderConfig(params=encoder_params),
    name='my_encoder'
)

model = create_model(config)
```

## Attention Types

### Softmax Attention
- **Complexity**: O(n²)
- **Use**: Short sequences, highest quality
```yaml
attention_type: softmax
```

### Linear Attention
- **Complexity**: O(n)
- **Use**: Long sequences, bidirectional
```yaml
attention_type: linear
nb_features: 256
```

### Causal Linear Attention
- **Complexity**: O(n)
- **Use**: Long sequences, autoregressive
```yaml
attention_type: causal_linear
nb_features: 256
```

## Activation Functions

Available activations:
- `GELU` - Gaussian Error Linear Unit
- `RELU` - Rectified Linear Unit
- `SILU` - Sigmoid Linear Unit (SwiSH)
- `GEGLU` - GELU with gating (GLU variant)
- `REGLU` - ReLU with gating (GLU variant)
- `SWIGLU` - SiLU with gating (GLU variant) ⭐ Recommended

```yaml
activation: SWIGLU  # Best performance
```

## Normalization Strategies

### LayerNorm (Default)
```yaml
use_rezero: false
use_scalenorm: false
```

### ReZero
Faster training with learnable scalar:
```yaml
use_rezero: true
```

### PreScaleNorm
L2 normalization with gain:
```yaml
use_scalenorm: true
```

## Example Configurations

See `configs/` directory for complete examples:
- `encoder_model.yml` - Encoder-only
- `decoder_model.yml` - Decoder-only
- `example_model.yml` - Encoder-decoder
- `full_experiment.yml` - Combined model + datasets

## Extending the Factory

### Add New Model Type

1. **Create model class** in `model/transformer.py`:
```python
class CustomTransformer(nn.Module):
    def __init__(self, **params):
        # ...
```

2. **Add config class** in `model/model_types.py`:
```python
@dataclass
class CustomConfig:
    params: CustomParams
    # ...
```

3. **Update ModelType**:
```python
ModelType = Literal["encoder", "decoder", "encoder_decoder", "custom"]
```

4. **Add factory method**:
```python
@staticmethod
def create_custom(config: CustomConfig) -> CustomTransformer:
    return CustomTransformer(**config.to_dict())
```

5. **Update create_model**:
```python
elif config.model_type == "custom":
    return ModelFactory.create_custom(config.config)
```

## Testing

Run comprehensive tests:
```bash
python test_model_factory.py
```

Tests include:
- ✅ All model type creation
- ✅ Combined parsing
- ✅ Factory methods
- ✅ Parameter counting
- ✅ Forward/backward passes
- ✅ Validation

## Summary

The model factory pattern provides:

📦 **Clean API** - Simple, consistent interface  
🔧 **Type Safety** - Catch errors at config time  
📝 **YAML Config** - Easy to read and modify  
🔄 **Reusability** - Share configs across experiments  
🎯 **Extensibility** - Add new models easily  
🤝 **Integration** - Works with dataset manager  

**Similar to data manager**, but for models! 🚀

