#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Normalization modules for transformer architectures.

This module implements pre-normalization wrappers that apply normalization
before the sublayer computation (attention or feed-forward). Pre-norm
architectures have been shown to stabilize training and enable deeper networks.

Includes:
1. PreScaleNorm: Scale normalization with learnable gain parameter
2. PreLayerNorm: Standard layer normalization wrapper

References:
    PreScaleNorm: Based on techniques from Performer architecture
                 https://github.com/lucidrains/performer-pytorch/blob/fc8b78441b1e27eb5d9b01fc738a8772cee07127/performer_pytorch/performer_pytorch.py

    PreLayerNorm: "Learning Deep Transformer Models for Machine Translation" (Wang et al., 2019)
                  Pre-norm shown to improve training stability
"""

from typing import Unpack

from torch import Tensor, nn, norm, ones

from .types import NormalizationParams


class PreScaleNorm(nn.Module):
    """
    Pre-normalization using scale normalization.

    Scale normalization normalizes the input to unit norm and applies a learnable
    gain parameter. This is a simpler alternative to layer normalization that can
    be more efficient while maintaining similar training dynamics.

    The normalization is applied BEFORE the sublayer (pre-norm architecture):
        output = fn(scale_norm(x))

    This approach:
    - Normalizes each feature vector to unit L2 norm
    - Applies a learnable scalar gain
    - More efficient than full layer normalization
    - Provides similar training stability benefits
    """

    def __init__(self, **kwargs: Unpack[NormalizationParams]):
        """
        Initialize PreScaleNorm wrapper.

        Args:
            kwargs: NormalizationParams
                dim: int - dimension of input features (typically d_model)
                fn: callable - the sublayer to wrap (attention or feed-forward)
                eps: float - epsilon for numerical stability (default 1e-5)
        """
        super().__init__()
        self.dim = kwargs["dim"]
        self.fn = kwargs["fn"]
        self.eps = kwargs.get("eps", 1e-5)

        # Learnable gain parameter
        self.g = nn.Parameter(ones(1))

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply scale normalization then the wrapped function.

        The input is normalized to unit L2 norm, scaled by learnable gain g,
        then passed to the wrapped function.

        Args:
            x: Tensor of shape (batch_size, seq_len, dim)
                Input tensor to normalize
            **kwargs: dict
                Additional arguments passed to the wrapped function
                (e.g., attention masks)

        Returns:
            Tensor: Output from the wrapped function after normalization
        """
        # Compute L2 norm along feature dimension
        n = norm(x, dim=-1, keepdim=True).clamp(min=self.eps)

        # Normalize and scale
        x = x / n * self.g

        # Apply wrapped function
        return self.fn(x, **kwargs)


class PreLayerNorm(nn.Module):
    """
    Pre-normalization using layer normalization.

    Standard layer normalization wrapper that applies LayerNorm before the
    sublayer computation. This is the most common pre-norm approach used in
    modern transformers.

    The normalization is applied BEFORE the sublayer (pre-norm architecture):
        output = fn(layer_norm(x))

    Pre-norm benefits:
    - More stable training, especially for deep networks
    - Reduces need for learning rate warmup
    - Allows training without careful initialization
    - Better gradient flow through deep networks
    """

    def __init__(self, **kwargs: Unpack[NormalizationParams]):
        """
        Initialize PreLayerNorm wrapper.

        Args:
            kwargs: NormalizationParams
                dim: int - dimension of input features (typically d_model)
                fn: callable - the sublayer to wrap (attention or feed-forward)
                eps: float - not used, kept for API consistency
        """
        super().__init__()
        self.dim = kwargs["dim"]
        self.fn = kwargs["fn"]

        # Standard layer normalization
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """
        Apply layer normalization then the wrapped function.

        The input is normalized using LayerNorm, then passed to the wrapped
        function. This is the standard pre-norm architecture.

        Args:
            x: Tensor of shape (batch_size, seq_len, dim)
                Input tensor to normalize
            **kwargs: dict
                Additional arguments passed to the wrapped function
                (e.g., attention masks)

        Returns:
            Tensor: Output from the wrapped function after normalization
        """
        # Apply layer norm then wrapped function
        return self.fn(self.norm(x), **kwargs)
