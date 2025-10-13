# Universal Encoder-Decoder Transformer Guide

## Overview

Your transformer implementation now supports **universal layers** that can be configured as:
- **Encoder-only** (bidirectional self-attention)
- **Decoder-only** (causal self-attention)
- **Encoder-Decoder** (full seq2seq with cross-attention)

## Architecture Components

### 1. PerformerLayer (Universal Layer)

The core building block that can function as either an encoder or decoder layer.

```python
from model.transformer import PerformerLayer
from model.activations import ActivationFunctions

# Encoder layer (no cross-attention)
encoder_layer = PerformerLayer(
    d_model=512,
    device=device,
    batch_size=32,
    num_heads=8,
    d_ff=4,
    dropout=0.1,
    attention_type="linear",           # or "softmax", "causal_linear"
    activation=ActivationFunctions.GELU,
    use_cross_attention=False,         # Encoder mode
)

# Decoder layer (with cross-attention)
decoder_layer = PerformerLayer(
    d_model=512,
    device=device,
    batch_size=32,
    num_heads=8,
    d_ff=4,
    dropout=0.1,
    attention_type="causal_linear",
    activation=ActivationFunctions.SWIGLU,
    use_cross_attention=True,          # Decoder mode
)
```

### 2. PerformerEncoder (Encoder-Only)

Bidirectional transformer for encoding source sequences.

```python
from model.transformer import PerformerEncoder

encoder = PerformerEncoder(
    d_model=512,
    device=device,
    batch_size=32,
    num_layers=6,
    num_heads=8,
    d_ff=4,
    dropout=0.1,
    attention_type="linear",           # Efficient O(n) attention
    activation=ActivationFunctions.GEGLU,
    nb_features=256,                   # For linear attention
)

# Encode source
src = torch.randn(32, 100, 512)       # (batch, seq_len, d_model)
encoded = encoder(src)
```

### 3. PerformerDecoder (Decoder with Cross-Attention)

Causal transformer with cross-attention for encoder-decoder models.

```python
from model.transformer import PerformerDecoder

decoder = PerformerDecoder(
    d_model=512,
    device=device,
    batch_size=32,
    num_layers=6,
    num_heads=8,
    d_ff=4,
    dropout=0.1,
    attention_type="causal_linear",    # Causal for autoregressive
    activation=ActivationFunctions.SWIGLU,
    nb_features=256,
    # use_cross_attention=True is default for PerformerDecoder
)

# Decode with encoder context
tgt = torch.randn(32, 50, 512)
encoder_output = torch.randn(32, 100, 512)
decoded = decoder(tgt, encoder_output=encoder_output)
```

### 4. EncoderDecoderTransformer (Full Seq2Seq)

Complete encoder-decoder architecture for sequence-to-sequence tasks.

```python
from model.transformer import EncoderDecoderTransformer

model = EncoderDecoderTransformer(
    encoder_params={
        "d_model": 512,
        "device": device,
        "batch_size": 32,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 4,
        "dropout": 0.1,
        "attention_type": "linear",
        "activation": ActivationFunctions.GEGLU,
        "nb_features": 256
    },
    decoder_params={
        "d_model": 512,
        "device": device,
        "batch_size": 32,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 4,
        "dropout": 0.1,
        "attention_type": "causal_linear",
        "activation": ActivationFunctions.SWIGLU,
        "nb_features": 256
    }
)

# Full forward pass
src = torch.randn(32, 100, 512)
tgt = torch.randn(32, 50, 512)
output = model(src, tgt)

# Or encode/decode separately
encoded = model.encode(src)
decoded = model.decode(tgt, encoder_output=encoded)
```

## Key Features

### 1. Attention Mechanisms

#### Softmax Attention (Standard)
- **Complexity**: O(n²)
- **Use**: Short sequences, highest quality
```python
attention_type="softmax"
```

#### Linear Attention
- **Complexity**: O(n)
- **Use**: Long sequences, bidirectional
```python
attention_type="linear"
nb_features=256  # Number of random features
```

#### Causal Linear Attention
- **Complexity**: O(n)
- **Use**: Long sequences, autoregressive (decoder)
```python
attention_type="causal_linear"
nb_features=256
```

### 2. Normalization Strategies

#### LayerNorm (Default)
Standard transformer pre-normalization.
```python
use_rezero=False
use_scalenorm=False
```

#### ReZero
Learnable scalar initialization, faster training.
```python
use_rezero=True
```

#### PreScaleNorm
L2 normalization with learnable gain.
```python
use_scalenorm=True
```

### 3. Activation Functions

GLU variants for better performance:
```python
from model.activations import ActivationFunctions

activation=ActivationFunctions.SWIGLU  # Recommended
activation=ActivationFunctions.GEGLU
activation=ActivationFunctions.REGLU
activation=ActivationFunctions.GELU
activation=ActivationFunctions.RELU
```

### 4. Masking

#### Self-Attention Masking
```python
# Padding mask
mask = torch.ones(batch_size, seq_len, seq_len)
mask[:, :, pad_positions] = 0

# Causal mask (automatic in decoder)
output = decoder(tgt, use_causal_mask=True)
```

#### Cross-Attention Masking
```python
# Control which source positions to attend to
cross_mask = torch.ones(batch_size, tgt_len, src_len)
cross_mask[:, :, pad_positions] = 0

output = model(src, tgt, cross_attn_mask=cross_mask)
```

## Use Cases

### 1. Machine Translation

```python
# Encoder: Bidirectional attention on source
# Decoder: Causal attention + cross-attention

translator = EncoderDecoderTransformer(
    encoder_params={
        "attention_type": "linear",  # Efficient for long sentences
        "activation": ActivationFunctions.GEGLU,
        # ... other params
    },
    decoder_params={
        "attention_type": "causal_linear",
        "activation": ActivationFunctions.SWIGLU,
        # ... other params
    }
)

# Training
src_embeddings = embed_source(src_tokens)
tgt_embeddings = embed_target(tgt_tokens)
output = translator(src_embeddings, tgt_embeddings)
logits = output_projection(output)
```

### 2. Text Summarization

Same architecture as translation:
```python
summarizer = EncoderDecoderTransformer(...)
document = embed(document_tokens)
summary_prefix = embed(summary_tokens)
summary_continuation = summarizer(document, summary_prefix)
```

### 3. Language Modeling (Decoder-Only)

```python
lm = PerformerDecoder(
    attention_type="causal_linear",
    use_cross_attention=False,  # No encoder
    # ... other params
)

# Autoregressive generation
tokens = embed(token_ids)
output = lm(tokens, use_causal_mask=True)
next_token_logits = output_projection(output[:, -1])
```

### 4. Text Classification (Encoder-Only)

```python
classifier = PerformerEncoder(
    attention_type="linear",
    # ... other params
)

# Encode and classify
text = embed(tokens)
encoded = classifier(text)
cls_token = encoded[:, 0]  # Use first token
logits = classification_head(cls_token)
```

## Performance Tips

### 1. Linear Attention for Long Sequences
When `seq_len > 1024`, use linear attention:
```python
attention_type="linear" or "causal_linear"
nb_features=min(256, d_head * log(d_head))  # As in Performer paper
```

### 2. Projection Matrix Redrawing
For training stability with linear attention:
```python
feature_redraw_interval=1000  # Redraw every 1000 steps
auto_check_redraw=True        # Automatic redrawing
```

Disable during inference:
```python
model.encoder.fix_projection_matrices_()
model.decoder.fix_projection_matrices_()
```

### 3. Memory Optimization
- Use `d_ff=4` (hidden_dim = 4 * d_model) as default
- For very long sequences, reduce to `d_ff=2`
- Use gradient checkpointing for very deep models

### 4. Recommended Configurations

#### Small Model (Fast)
```python
d_model=256
num_layers=4
num_heads=4
d_ff=4
attention_type="linear"
```

#### Medium Model (Balanced)
```python
d_model=512
num_layers=6
num_heads=8
d_ff=4
attention_type="linear"
```

#### Large Model (Quality)
```python
d_model=1024
num_layers=12
num_heads=16
d_ff=4
attention_type="causal_linear"  # For decoder
```

## Advanced Usage

### Custom Forward Pass
```python
# Manual encoding
encoded = model.encoder(src, mask=src_mask)

# Manual decoding with custom masks
decoded = model.decoder(
    tgt,
    encoder_output=encoded,
    self_attn_mask=causal_mask,
    cross_attn_mask=cross_mask,
    use_causal_mask=False  # Using custom mask
)
```

### Mixed Attention Types
```python
# Encoder with softmax, Decoder with linear
model = EncoderDecoderTransformer(
    encoder_params={"attention_type": "softmax"},
    decoder_params={"attention_type": "causal_linear", "nb_features": 256}
)
```

### Different Model Sizes
```python
# Asymmetric encoder-decoder
model = EncoderDecoderTransformer(
    encoder_params={
        "d_model": 512,
        "num_layers": 12,  # Deeper encoder
    },
    decoder_params={
        "d_model": 512,
        "num_layers": 6,   # Shallower decoder
    }
)
```

## Testing

Run comprehensive tests:
```bash
python test_encoder_decoder.py
```

This tests:
- ✅ Universal layer (encoder and decoder modes)
- ✅ Encoder-only models
- ✅ Decoder with cross-attention
- ✅ Full encoder-decoder seq2seq
- ✅ Different configurations (normalization, activation)
- ✅ Masking scenarios
- ✅ Gradient flow

## Summary

You now have a **complete, universal transformer** implementation:

📦 **Components**:
- `PerformerLayer` - Universal layer
- `PerformerEncoder` - Encoder-only
- `PerformerDecoder` - Decoder with cross-attention
- `EncoderDecoderTransformer` - Full seq2seq

⚡ **Features**:
- Self-attention and cross-attention
- Linear attention (O(n) complexity)
- Causal and bidirectional variants
- Multiple normalization options
- GLU activation variants
- Flexible masking

🎯 **Ready for**:
- Machine Translation
- Text Summarization
- Language Modeling
- Text Classification
- Any sequence-to-sequence task

Happy building! 🚀

