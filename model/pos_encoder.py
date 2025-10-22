#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Positional encoding modules for transformers.

This module implements various positional encoding strategies to inject
position information into transformer models:

1. RoPE (Rotary Position Embedding): Relative position encoding via rotation
2. Sinusoidal: Fixed sinusoidal encoding (original Transformer)
3. Learnable: Learned absolute position embeddings
4. None: No positional encoding (for tasks that don't need it)

References:
    RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2023)
          https://arxiv.org/abs/2104.09864

    Sinusoidal: "Attention Is All You Need" (Vaswani et al., 2017)
                https://arxiv.org/abs/1706.03762
"""

import math
from typing import Tuple
from typing_extensions import Unpack

import torch
from torch import Tensor, nn

from .types import PositionalEncodingParams


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    RoPE encodes position information by rotating the query and key vectors
    in the attention mechanism. Unlike absolute positional encodings, RoPE
    naturally captures relative position information and has better extrapolation
    to longer sequences.

    The key idea is to multiply the features by rotation matrices whose angles
    are proportional to the position, creating a natural decay of correlation
    with distance.
    """

    def __init__(self, **kwargs: Unpack[PositionalEncodingParams]):
        """
        Initialize RoPE.

        Args:
            kwargs: PositionalEncodingParams
                encoding_type: PositionalEncodingType - should be "rope"
                max_seq_len: int - maximum sequence length to precompute frequencies for
                d_model: int - dimension of the embeddings (should be even).
                    Typically the head dimension (d_model / num_heads)
                device: torch.device - device for computation
                theta: float - base value for frequency computation. Default 10000.0
        """
        super().__init__()

        dim = kwargs["d_model"]
        max_seq_len = kwargs["max_seq_len"]
        theta = kwargs.get("theta", 10000.0)
        device = kwargs["device"]

        assert dim % 2 == 0, "RoPE requires even dimension"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for efficiency
        self._compute_cos_sin_cache(max_seq_len, device)

    def _compute_cos_sin_cache(
        self, seq_len: int, device: torch.device | None = None
    ) -> None:
        """Precompute and cache cos/sin values for all positions."""
        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # Compute all frequencies: outer product [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Create [seq_len, dim] by repeating each frequency
        emb = torch.cat([freqs, freqs], dim=-1)

        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.cached_seq_len = seq_len

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary position embedding to queries and keys.

        Args:
            q: Tensor of shape (..., seq_len, dim)
                Query tensor
            k: Tensor of shape (..., seq_len, dim)
                Key tensor

        Returns:
            Tuple[Tensor, Tensor]: Rotated (q, k) tensors with same shape
        """
        seq_len = q.shape[-2]

        # Extend cache if needed
        if seq_len > self.cached_seq_len:
            self._compute_cos_sin_cache(seq_len, device=q.device)

        # Get cos and sin for current sequence length
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]

        # Apply rotation: x * cos + rotate_half(x) * sin
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from the original Transformer paper.

    Uses sine and cosine functions of different frequencies to encode
    absolute positions. The encoding is fixed (not learned) and allows
    the model to attend to relative positions via linear transformations.
    """

    def __init__(self, **kwargs: Unpack[PositionalEncodingParams]):
        """
        Initialize sinusoidal positional encoding.

        Args:
            kwargs: PositionalEncodingParams
                encoding_type: PositionalEncodingType - should be "sinusoidal"
                max_seq_len: int - maximum sequence length to precompute
                d_model: int - dimension of the model embeddings
                device: torch.device - device for computation
                theta: float - base value for frequency computation. Default 10000.0
        """
        super().__init__()

        d_model = kwargs["d_model"]
        max_seq_len = kwargs["max_seq_len"]
        theta = kwargs.get("theta", 10000.0)
        device = kwargs["device"]

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Create positional encoding matrix
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(theta) / d_model))

        pe = torch.zeros(max_seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
                Input embeddings

        Returns:
            Tensor: Input with added positional encoding, same shape as input
        """
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        )

        # Add positional encoding (broadcasting over batch dimension)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Absolute Positional Encoding.

    Position embeddings are learned during training. This is simpler than
    sinusoidal encoding and can potentially learn task-specific position
    representations.
    """

    def __init__(self, **kwargs: Unpack[PositionalEncodingParams]):
        """
        Initialize learnable positional encoding.

        Args:
            kwargs: PositionalEncodingParams
                encoding_type: PositionalEncodingType - should be "learnable"
                max_seq_len: int - maximum sequence length (vocabulary size for positions)
                d_model: int - dimension of the model embeddings
                device: torch.device - device for computation
                theta: float - not used for learnable encoding
        """
        super().__init__()

        d_model = kwargs["d_model"]
        max_seq_len = kwargs["max_seq_len"]
        device = kwargs["device"]

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_seq_len, d_model, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add learnable positional encoding to input.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
                Input embeddings

        Returns:
            Tensor: Input with added positional encoding, same shape as input
        """
        batch_size, seq_len, _ = x.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        )

        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)

        # Get position embeddings and add to input
        pos_emb = self.position_embeddings(positions)
        return x + pos_emb


class PositionalEncodingFactory:
    """
    Factory class for creating positional encoding modules.

    Provides a unified interface for creating different types of positional
    encodings based on configuration.
    """

    @staticmethod
    def create(**kwargs: Unpack[PositionalEncodingParams]) -> nn.Module | None:
        """
        Create a positional encoding module.

        Args:
            kwargs: PositionalEncodingParams
                encoding_type: PositionalEncodingType - type of encoding
                d_model: int - model dimension
                max_seq_len: int - maximum sequence length
                device: torch.device - device for computation
                theta: float - base for frequency computation (default 10000.0)

        Returns:
            nn.Module | None: The positional encoding module, or None if type is "none"

        Raises:
            ValueError: If encoding_type is not recognized
        """
        encoding_type = kwargs["encoding_type"]

        if encoding_type == "rope":
            return RotaryPositionEmbedding(**kwargs)
        elif encoding_type == "sinusoidal":
            return SinusoidalPositionalEncoding(**kwargs)
        elif encoding_type == "learnable":
            return LearnablePositionalEncoding(**kwargs)
        elif encoding_type == "none":
            return None
        else:
            raise ValueError(
                f"Unknown positional encoding type: {encoding_type}. "
                f"Choose from: 'rope', 'sinusoidal', 'learnable', 'none'"
            )


# Helper functions for applying RoPE to attention


def apply_rotary_emb(
    q: Tensor,
    k: Tensor,
    rope: RotaryPositionEmbedding,
) -> Tuple[Tensor, Tensor]:
    """
    Helper function to apply RoPE to query and key tensors.

    Args:
        q: Tensor of shape (batch, heads, seq_len, head_dim)
            Query tensor
        k: Tensor of shape (batch, heads, seq_len, head_dim)
            Key tensor
        rope: RotaryPositionEmbedding
            RoPE module instance

    Returns:
        Tuple[Tensor, Tensor]: Rotated (q, k) tensors
    """
    return rope(q, k)
