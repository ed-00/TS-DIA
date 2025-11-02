#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Model package for transformer architectures.

This package provides transformer models with various attention mechanisms
including softmax attention and linear attention (Performer).
"""

from .activations import ActivationFunctions
from .attention import MultiHeadAttention
from .feedforward import FeedForward, ReZero
from .model_factory import ModelFactory, create_model
from .model_types import (
    DecoderConfig,
    EncoderConfig,
    EncoderDecoderConfig,
    ModelConfig,
)
from .norm import PreLayerNorm, PreScaleNorm
from .transformer import (
    EncoderDecoderTransformer,
    Performer,
    PerformerDecoder,
    PerformerEncoder,
    PerformerLayer,
)
from .types import (
    AttentionType,
    FeedForwardParams,
    MultiHeadAttentionParams,
    PerformerLayerParams,
    PerformerParams,
    PositionalEncodingType,
    TransformerSharedParams,
)
from .utils import ProjectionUpdater

__all__ = [
    # Activation functions
    "ActivationFunctions",
    # Attention
    "MultiHeadAttention",
    # Feed-forward
    "FeedForward",
    "ReZero",
    # Model factory
    "ModelFactory",
    "create_model",
    # Model configurations
    "DecoderConfig",
    "EncoderConfig",
    "EncoderDecoderConfig",
    "ModelConfig",
    # Normalization
    "PreLayerNorm",
    "PreScaleNorm",
    # Transformer models
    "EncoderDecoderTransformer",
    "Performer",
    "PerformerDecoder",
    "PerformerEncoder",
    "PerformerLayer",
    # Types
    "AttentionType",
    "FeedForwardParams",
    "MultiHeadAttentionParams",
    "PerformerLayerParams",
    "PerformerParams",
    "PositionalEncodingType",
    "TransformerSharedParams",
    # Utils
    "ProjectionUpdater",
]

