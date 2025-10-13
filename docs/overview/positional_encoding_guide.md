# Positional Encoding Guide

This guide explains the positional encoding implementations available in the TS-DIA transformer model and how to use them effectively.

## Overview

Positional encoding is essential for transformer models to understand the order of tokens in a sequence. Since attention mechanisms are permutation-invariant, we need to inject position information explicitly.

## Available Methods

### 1. RoPE (Rotary Position Embedding) ⭐ Recommended

**Best for**: Long sequences, relative position modeling, modern transformers

RoPE (Rotary Position Embedding) encodes positions by rotating query and key vectors. This creates a natural relative position encoding that:
- Works excellently for long sequences
- Has better extrapolation to unseen sequence lengths
- Captures relative positions naturally
- Used in modern models like LLaMA, PaLM, GPT-NeoX

**How it works**:
- Applied **within attention** to Q and K vectors
- Rotates features by an angle proportional to position
- Distance between positions is preserved through rotation

**Usage**:
```python
from model.pos_encoder import RotaryPositionEmbedding

# Create RoPE (use head_dim, not d_model!)
head_dim = d_model // num_heads
rope = RotaryPositionEmbedding(
    dim=head_dim,
    max_seq_len=2048,
    theta=10000.0,
    device=device
)

# Apply to Q and K in attention
q_rot, k_rot = rope(q, k)  # q, k: (batch, heads, seq_len, head_dim)
```

**References**:
- Paper: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

---

### 2. Sinusoidal Positional Encoding

**Best for**: Classic transformers, interpretability, fixed encoding

Sinusoidal encoding uses sine and cosine functions at different frequencies. This is the original method from "Attention Is All You Need".

**How it works**:
- Uses fixed sine/cosine functions
- Added to input embeddings **before attention**
- Allows model to learn relative positions via linear transformations

**Usage**:
```python
from model.pos_encoder import SinusoidalPositionalEncoding

pos_encoder = SinusoidalPositionalEncoding(
    d_model=512,
    max_seq_len=5000,
    theta=10000.0,
    device=device
)

# Add to embeddings
x_with_pos = pos_encoder(token_embeddings)  # (batch, seq_len, d_model)
```

**References**:
- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

### 3. Learnable Positional Encoding

**Best for**: Task-specific optimization, when you have enough data

Learnable position embeddings are optimized during training, allowing the model to learn task-specific position representations.

**How it works**:
- Position embeddings are parameters (learned)
- Added to input embeddings **before attention**
- Can learn complex position patterns specific to your task

**Usage**:
```python
from model.pos_encoder import LearnablePositionalEncoding

pos_encoder = LearnablePositionalEncoding(
    d_model=512,
    max_seq_len=5000,
    device=device
)

# Add to embeddings (will be optimized during training)
x_with_pos = pos_encoder(token_embeddings)  # (batch, seq_len, d_model)
```

---

### 4. No Positional Encoding

**Best for**: Sets, bags-of-words, position-invariant tasks

Use `encoding_type="none"` when position information is not needed.

---

## Factory Pattern

Use `PositionalEncodingFactory` for configuration-based creation:

```python
from model.pos_encoder import PositionalEncodingFactory

# From config
pos_encoder = PositionalEncodingFactory.create(
    encoding_type="rope",  # or "sinusoidal", "learnable", "none"
    d_model=512,
    max_seq_len=2048,
    theta=10000.0,
    device=device
)
```

---

## Comparison

| Method | Applied | Complexity | Extrapolation | Learned | Best Use Case |
|--------|---------|-----------|---------------|---------|---------------|
| **RoPE** | In-attention (Q,K) | O(d) | Excellent | No | Long sequences, relative positions |
| **Sinusoidal** | Pre-attention | O(1) | Good | No | Classic transformers, interpretability |
| **Learnable** | Pre-attention | O(1) | Poor | Yes | Task-specific, enough data |
| **None** | - | - | - | - | Position-invariant tasks |

---

## Integration with Attention

### RoPE Integration

RoPE is applied **within** the attention mechanism:

```python
class AttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, device):
        super().__init__()
        self.head_dim = d_model // num_heads
        
        # Create RoPE
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,  # ← Important: use head_dim
            max_seq_len=max_seq_len,
            device=device
        )
        
        # Q, K, V projections...
        self.W_q = nn.Linear(d_model, d_model, device=device)
        self.W_k = nn.Linear(d_model, d_model, device=device)
        self.W_v = nn.Linear(d_model, d_model, device=device)
        
    def forward(self, x):
        # Project and reshape
        q = self.W_q(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Compute attention...
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        # ... rest of attention
```

### Sinusoidal/Learnable Integration

These are applied **before** attention:

```python
class TransformerWithPosEnc(nn.Module):
    def __init__(self, d_model, max_seq_len, device):
        super().__init__()
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            device=device
        )
        
        # Attention layer
        self.attention = MultiHeadAttention(...)
        
    def forward(self, x):
        # Add positional encoding first
        x = self.pos_encoder(x)
        
        # Then apply attention
        x = self.attention(x)
        return x
```

---

## Configuration

Add to your model config:

```python
# In types.py
from model.types import PositionalEncodingParams

pos_config = {
    "encoding_type": "rope",  # "rope", "sinusoidal", "learnable", or "none"
    "max_seq_len": 2048,
    "d_model": 512,
    "device": device,
    "theta": 10000.0,
}
```

---

## Tips & Best Practices

### When to use RoPE:
- ✅ Building modern transformers
- ✅ Long sequence modeling
- ✅ Need good extrapolation
- ✅ Relative position is important

### When to use Sinusoidal:
- ✅ Classic Transformer architecture
- ✅ Want interpretable positions
- ✅ Don't want to train position params
- ✅ Good baseline

### When to use Learnable:
- ✅ Task-specific optimization
- ✅ Have enough training data
- ✅ Position patterns are complex
- ⚠️ May not generalize to longer sequences

### Important Notes:

1. **RoPE dimension**: Always use `head_dim = d_model // num_heads`, NOT `d_model`
2. **Max sequence length**: Set higher than your longest expected sequence
3. **Theta parameter**: 10000.0 is standard, higher values = slower frequency change
4. **Gradient flow**: RoPE is differentiable, gradients flow through rotations

---

## Examples

See `examples/positional_encoding_usage.py` for complete working examples of all methods.

---

## References

1. **RoPE**: Su et al. (2023) - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
2. **Sinusoidal**: Vaswani et al. (2017) - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. **Implementation Reference**: [performer-pytorch by lucidrains](https://github.com/lucidrains/performer-pytorch)

