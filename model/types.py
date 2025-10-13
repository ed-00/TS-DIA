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
from typing import TypedDict

from torch import device

from .activations import ActivationFunctions


@dataclass
class TransformerSharedParams(TypedDict):
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

    Extends TransformerSharedParams with attention-specific configuration.
    Used to initialize MultiHeadAttention and CrossAttention modules.

    Attributes:
        num_heads: int
            Number of attention heads. The model dimension (d_model) must be
            divisible by num_heads. Each head operates on d_model/num_heads dimensions.
        dropout: float
            Dropout rate applied to attention weights and output for regularization.
            Value should be between 0.0 and 1.0.
    """

    num_heads: int
    dropout: float


@dataclass
class FeedForwardParams(TransformerSharedParams):
    """
    Parameters for feed-forward network modules.

    Extends TransformerSharedParams with feed-forward specific configuration.
    Used to initialize FeedForward modules in transformer blocks.

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
