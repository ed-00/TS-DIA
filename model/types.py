#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Type definitions for transformer components.

This module defines the parameter structures used across various transformer
modules including multi-head attention and feed-forward networks. It provides
typed dictionaries that ensure type safety and clear parameter specifications
for model initialization.
"""

from dataclasses import dataclass
from typing import Literal

from torch import device

from .activations import ActivationFunctions

# Attention type options
AttentionType = Literal["softmax", "linear", "causal_linear"]

# Positional encoding type options
PositionalEncodingType = Literal["rope", "sinusoidal", "learnable", "none"]


@dataclass
class TransformerSharedParams:
    """
    Shared parameters for all transformer modules.

    This base class defines common parameters that are used across all
    transformer components, ensuring consistency in model configuration.

    Attributes:
        d_model: int
            The dimension of the model's embeddings and hidden states.
            This is the primary dimensionality used throughout the transformer.
        device: torch.device
            The device (CPU/GPU) on which the model will be executed.
            Determines where tensors are allocated and computations occur.
        batch_size: int
            The batch size for processing sequences. Used for reshaping
            operations and memory allocation in attention mechanisms.
    """

    d_model: int
    device: device
    batch_size: int


@dataclass
class MultiHeadAttentionParams(TransformerSharedParams):
    """
    Parameters for multi-head attention modules.

    Inherits: d_model, device, batch_size from TransformerSharedParams

    Attributes:
        num_heads: int
            Number of attention heads. The model dimension (d_model) must be
            divisible by num_heads. Each head operates on d_model/num_heads dimensions.
        dropout: float
            Dropout rate applied to attention weights and output for regularization.
            Value should be between 0.0 and 1.0.
        attention_type: AttentionType
            Type of attention mechanism to use. Options:
            - "softmax": Standard scaled dot-product attention (O(n²) complexity)
            - "linear": Linear attention with kernel approximation (O(n) complexity)
            - "causal_linear": Causal linear attention for autoregressive tasks
        nb_features: int | None
            Number of random features for linear attention kernel approximation.
            If None, defaults to d_head * log(d_head). Only used when attention_type is "linear" or "causal_linear".
    """

    num_heads: int
    dropout: float
    attention_type: AttentionType = "softmax"
    nb_features: int | None = None


@dataclass
class FeedForwardParams(TransformerSharedParams):
    """
    Parameters for feed-forward network modules.

    Inherits: d_model, device, batch_size from TransformerSharedParams

    Attributes:
        activation: ActivationFunctions
            The activation function to use in the feed-forward network.
            Supports both standard activations (ReLU, GELU) and GLU variants
            (SwiGLU, GeGLU) for improved performance.
        d_ff: int
            Expansion factor for the hidden dimension in the feed-forward network.
            The intermediate dimension is d_model * d_ff. Common values are 4 or 8.
        dropout: float
            Dropout rate applied after activation for regularization.
            Value should be between 0.0 and 1.0.
    """

    activation: ActivationFunctions
    d_ff: int
    dropout: float


@dataclass
class PositionalEncodingParams(TransformerSharedParams):
    """
    Parameters for positional encoding modules.

    Inherits: d_model, device, batch_size from TransformerSharedParams

    Attributes:
        encoding_type: PositionalEncodingType
            Type of positional encoding to use:
            - "rope": Rotary Position Embedding (RoPE) - relative position encoding
            - "sinusoidal": Fixed sinusoidal encoding from original Transformer
            - "learnable": Learned absolute position embeddings
            - "none": No positional encoding
        max_seq_len: int
            Maximum sequence length supported by the positional encoding.
            For RoPE, this determines the frequency range.
            For learnable/sinusoidal, this is the vocabulary size.
        theta: float
            Base value for frequency computation in RoPE and sinusoidal encodings.
            Default is 10000.0 (from original Transformer paper).
    """

    encoding_type: PositionalEncodingType
    max_seq_len: int
    theta: float = 10000.0


@dataclass
class NormalizationParams:
    """
    Parameters for normalization wrapper modules.

    Defines configuration for normalization techniques that wrap transformer
    sublayers (attention and feed-forward networks). These normalizations are
    applied before the sublayer computation (pre-norm architecture).

    Attributes:
        dim: int
            Dimension of the input features to normalize. Typically d_model.
        fn: callable
            The function or module to wrap with normalization. This is usually
            an attention layer or feed-forward network that will be called
            after normalization is applied.
        eps: float
            Small epsilon value for numerical stability in normalization.
            Only used for PreScaleNorm. Default is 1e-5.
    """

    dim: int
    fn: callable
    eps: float = 1e-5


@dataclass
class PerformerLayerParams(TransformerSharedParams):
    """
    Parameters for a single Performer transformer layer.

    Universal layer that can be used as:
    - Encoder layer (self-attention only)
    - Decoder layer (causal self-attention + cross-attention)

    Inherits: d_model, device, batch_size from TransformerSharedParams

    Attributes:
        num_heads: int
            Number of attention heads
        d_ff: int
            Feed-forward expansion factor (hidden dim = d_model * d_ff)
        dropout: float
            Dropout rate for attention and feed-forward
        attention_type: AttentionType
            Type of attention: "softmax", "linear", or "causal_linear"
        activation: ActivationFunctions
            Activation function for feed-forward network
        use_cross_attention: bool
            If True, add cross-attention sublayer (for decoder). Default False.
        use_rezero: bool
            If True, use ReZero instead of LayerNorm. Default False.
        use_scalenorm: bool
            If True, use PreScaleNorm instead of LayerNorm. Default False.
        nb_features: int | None
            Number of random features for linear attention. Default None (auto).
    """

    num_heads: int
    d_ff: int
    dropout: float
    attention_type: AttentionType = "softmax"
    activation: ActivationFunctions = None
    use_cross_attention: bool = False
    use_rezero: bool = False
    use_scalenorm: bool = False
    nb_features: int | None = None


@dataclass
class PerformerParams(TransformerSharedParams):
    """
    Parameters for complete Performer model (encoder, decoder, or encoder-decoder).

    Stacks multiple Performer layers into a complete transformer.
    Can be used for encoder-only, decoder-only, or encoder-decoder architectures.

    Inherits: d_model, device, batch_size from TransformerSharedParams

    Attributes:
        num_layers: int
            Number of transformer layers to stack
        num_heads: int
            Number of attention heads per layer
        d_ff: int
            Feed-forward expansion factor
        dropout: float
            Dropout rate for attention and feed-forward
        attention_type: AttentionType
            Type of attention mechanism
        activation: ActivationFunctions
            Activation function for feed-forward
        use_cross_attention: bool
            Add cross-attention sublayers (for decoder). Default False.
        use_rezero: bool
            Use ReZero normalization. Default False.
        use_scalenorm: bool
            Use PreScaleNorm normalization. Default False.
        nb_features: int | None
            Number of random features for linear attention
        feature_redraw_interval: int | None
            Redraw projections every N iterations. None = never redraw.
        auto_check_redraw: bool
            Automatically check and redraw projections. Default True.
        num_classes: int | None
            Number of output classes for classification tasks. If specified,
            adds a linear projection layer: d_model -> num_classes.
            Used for sequence/token classification (e.g., diarization, NER).
            Default None (no output projection).
    """

    num_layers: int
    num_heads: int
    d_ff: int
    dropout: float
    attention_type: AttentionType = "softmax"
    activation: ActivationFunctions = None
    use_cross_attention: bool = False
    use_rezero: bool = False
    use_scalenorm: bool = False
    nb_features: int | None = None
    feature_redraw_interval: int | None = 1000
    auto_check_redraw: bool = True
    num_classes: int | None = None
