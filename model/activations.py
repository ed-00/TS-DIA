#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Activation functions module.

This module provides various activation functions including gated linear units (GLU variants).
GLU variants split the input into two halves, apply an activation function to one half,
and multiply it with the other half for improved expressiveness in neural networks.
"""

from enum import Enum
from torch import Tensor, nn


class GeGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation.

    This activation function splits the input tensor into two halves along the last dimension,
    applies GELU activation to the first half, and multiplies it element-wise with the second half.
    This gating mechanism allows the network to control information flow more effectively.
    """

    def __init__(self) -> None:
        """Initialize GeGLU activation function."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply GeGLU activation.

        Args:
            x: Tensor of shape (..., 2 * hidden_dim) where the last dimension will be split in half.

        Returns:
            Tensor of shape (..., hidden_dim) after applying GELU(a) * b where a and b are the two halves.
        """
        a, b = x.chunk(2, dim=-1)
        return nn.functional.gelu(a) * b


class ReGLU(nn.Module):
    """
    Gated Linear Unit with ReLU activation.

    This activation function splits the input tensor into two halves along the last dimension,
    applies ReLU activation to the first half, and multiplies it element-wise with the second half.
    """

    def __init__(self) -> None:
        """Initialize ReGLU activation function."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReGLU activation.

        Args:
            x: Tensor of shape (..., 2 * hidden_dim) where the last dimension will be split in half.

        Returns:
            Tensor of shape (..., hidden_dim) after applying ReLU(a) * b where a and b are the two halves.
        """
        a, b = x.chunk(2, dim=-1)
        return nn.functional.relu(a) * b


class SwiGLU(nn.Module):
    """
    Gated Linear Unit with SiLU (Swish) activation.

    This activation function splits the input tensor into two halves along the last dimension,
    applies SiLU (also known as Swish) activation to the first half, and multiplies it element-wise
    with the second half. SwiGLU has been shown to improve performance in transformer models.
    """

    def __init__(self) -> None:
        """Initialize SwiGLU activation function."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU activation.

        Args:
            x: Tensor of shape (..., 2 * hidden_dim) where the last dimension will be split in half.

        Returns:
            Tensor of shape (..., hidden_dim) after applying SiLU(a) * b where a and b are the two halves.
        """
        a, b = x.chunk(2, dim=-1)
        return nn.functional.silu(a) * b


class ActivationFunctions(Enum):
    """
    Enumeration of available activation functions.

    This enum provides a convenient way to select activation functions by name.
    Each member maps to the corresponding activation function class.

    Usage:
        activation = ActivationFunctions.GEGLU.value()
        output = activation(input_tensor)
    """
    RELU = nn.ReLU
    SILU = nn.SiLU
    GELU = nn.GELU
    GEGLU = GeGLU
    REGLU = ReGLU
    SWIGLU = SwiGLU


def _is_glu(activation: ActivationFunctions):
    if ActivationFunctions.RELU:
        return False
    return True

