# Normalization Guide

This guide explains the normalization techniques available in the TS-DIA transformer implementation and when to use them.

## Overview

Normalization is crucial for stable training of deep neural networks. This module implements **pre-normalization** wrappers that apply normalization **before** the sublayer computation (attention or feed-forward network).

Pre-norm architecture has become the standard in modern transformers due to better training stability.

## Pre-Norm vs Post-Norm

**Pre-Norm (Recommended):**
```
output = x + sublayer(normalize(x))
```
- Normalization applied **before** the sublayer
- More stable training, especially for deep networks
- Reduces need for learning rate warmup
- Better gradient flow

**Post-Norm (Original Transformer):**
```
output = normalize(x + sublayer(x))
```
- Normalization applied **after** the residual addition
- Requires careful initialization and warmup
- Can be unstable for very deep networks

## Available Normalization Methods

### 1. PreLayerNorm ⭐ Recommended

**Best for**: Most transformer architectures, standard approach

Standard layer normalization applied before the sublayer. This is the most common pre-norm technique used in modern transformers.

**How it works**:
- Normalizes features to zero mean and unit variance
- Applies learnable affine transformation (scale and shift)
- Most widely used and well-tested approach

**Usage**:
```python
from model.norm import PreLayerNorm
from model.attention import MultiHeadAttention

# Create attention layer
attention = MultiHeadAttention(
    device=device,
    d_model=512,
    num_heads=8,
    dropout=0.1,
    batch_size=32
)

# Wrap with PreLayerNorm
norm_attention = PreLayerNorm(
    dim=512,
    fn=attention
)

# Use in forward pass
output = norm_attention(x)  # Applies LayerNorm then attention
```

---

### 2. PreScaleNorm

**Best for**: Efficiency, when you want lighter normalization

Scale normalization with learnable gain parameter. Simpler and more efficient alternative to LayerNorm.

**How it works**:
- Normalizes each feature vector to unit L2 norm
- Applies a single learnable scalar gain parameter
- More efficient than full layer normalization

**Usage**:
```python
from model.norm import PreScaleNorm
from model.feedforward import FeedForward

# Create feed-forward layer
ff = FeedForward(
    d_model=512,
    d_ff=2048,
    dropout=0.1,
    activation=ActivationFunctions.GELU
)

# Wrap with PreScaleNorm
norm_ff = PreScaleNorm(
    dim=512,
    fn=ff,
    eps=1e-5
)

# Use in forward pass
output = norm_ff(x)  # Applies scale norm then feed-forward
```

---

## Integration with Transformer Blocks

### Standard Transformer Block with Pre-Norm

```python
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.norm import PreLayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Self-attention with pre-norm
        self.attn = MultiHeadAttention(
            device=device,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_size=32
        )
        self.norm_attn = PreLayerNorm(dim=d_model, fn=self.attn)
        
        # Feed-forward with pre-norm
        self.ff = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=ActivationFunctions.GELU
        )
        self.norm_ff = PreLayerNorm(dim=d_model, fn=self.ff)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        x = x + self.norm_attn(x, mask=mask)
        
        # Feed-forward with residual
        x = x + self.norm_ff(x)
        
        return x
```

### Using with ReZero (Alternative)

ReZero can be combined with normalization or used instead:

```python
from model.feedforward import ReZero
from model.norm import PreLayerNorm

# Option 1: ReZero + PreLayerNorm
norm_attn = ReZero(PreLayerNorm(dim=512, fn=attention))

# Option 2: Just ReZero (no normalization)
rezero_attn = ReZero(attention)
```

---

## Comparison

| Method | Params | Computation | Stability | When to Use |
|--------|--------|-------------|-----------|-------------|
| **PreLayerNorm** | 2×d_model | O(d) | Excellent | Default choice, proven approach |
| **PreScaleNorm** | 1 | O(d) | Good | Efficiency, lighter models |

**PreLayerNorm Benefits:**
- ✅ Most widely used and tested
- ✅ Excellent training stability
- ✅ Works well for all model sizes
- ✅ Well-understood properties

**PreScaleNorm Benefits:**
- ✅ More efficient (fewer parameters)
- ✅ Faster computation
- ✅ Good for memory-constrained settings
- ✅ Based on Performer architecture

---

## Configuration

### Using TypedDict Parameters

Both normalization classes support the `**kwargs: Unpack[NormalizationParams]` pattern:

```python
from model.types import NormalizationParams

# Configuration dictionary
norm_config: NormalizationParams = {
    "dim": 512,
    "fn": attention_layer,
    "eps": 1e-5,  # Optional, only for PreScaleNorm
}

# Create normalization
norm_layer = PreLayerNorm(**norm_config)
```

---

## Best Practices

### 1. Use Pre-Norm for Deep Networks
Pre-normalization enables training of much deeper networks without careful initialization:

```python
# Deep transformer (e.g., 24+ layers)
for i in range(num_layers):
    x = x + PreLayerNorm(dim=d_model, fn=attention_layers[i])(x)
    x = x + PreLayerNorm(dim=d_model, fn=ff_layers[i])(x)
```

### 2. Choose Normalization Based on Constraints

**Use PreLayerNorm when:**
- You want the standard, well-tested approach
- Training stability is critical
- You have sufficient compute/memory

**Use PreScaleNorm when:**
- You need efficiency
- Working with resource constraints
- Building on Performer-style architectures

### 3. Epsilon for Numerical Stability

For PreScaleNorm, the epsilon parameter prevents division by zero:

```python
# Default (recommended)
norm = PreScaleNorm(dim=512, fn=layer, eps=1e-5)

# For half-precision training, use larger epsilon
norm = PreScaleNorm(dim=512, fn=layer, eps=1e-3)
```

---

## Implementation Details

### PreLayerNorm

Uses PyTorch's built-in `nn.LayerNorm`:
- Normalizes across the feature dimension
- Learnable affine parameters (scale γ, shift β)
- Formula: `γ * (x - μ) / σ + β`

### PreScaleNorm

Custom implementation:
- L2 normalization to unit norm
- Single learnable gain parameter `g`
- Formula: `g * x / ||x||₂`

---

## References

1. **Pre-Norm Architecture**: Wang et al. (2019) - [Learning Deep Transformer Models for Machine Translation](https://arxiv.org/abs/1906.01787)

2. **PreScaleNorm**: Adapted from Performer architecture
   - GitHub: [performer-pytorch by lucidrains](https://github.com/lucidrains/performer-pytorch/blob/fc8b78441b1e27eb5d9b01fc738a8772cee07127/performer_pytorch/performer_pytorch.py)

3. **Layer Normalization**: Ba et al. (2016) - [Layer Normalization](https://arxiv.org/abs/1607.06450)

---

## Examples

See the module docstrings in `model/norm.py` for additional usage examples and implementation details.

### Quick Start

```python
from model.norm import PreLayerNorm, PreScaleNorm
from model.attention import MultiHeadAttention

# Create your sublayer
attention = MultiHeadAttention(...)

# Wrap with normalization (choose one)
norm_attention = PreLayerNorm(dim=512, fn=attention)  # Standard
# OR
norm_attention = PreScaleNorm(dim=512, fn=attention)  # Efficient

# Use in forward pass
output = norm_attention(input_tensor)
```
