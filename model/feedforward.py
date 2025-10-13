#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Feed-forward network and normalization modules.

This module implements:
1. FeedForward: A position-wise feed-forward network that is applied to each
   position separately and identically. It consists of two linear transformations
   with an activation function in between, and supports both standard activations
   and gated linear units (GLU) variants.

2. ReZero: A normalization technique that replaces traditional layer normalization
   with a learnable scalar parameter, enabling faster convergence and deeper
   network training.
"""

from typing import Unpack

from torch import Tensor, nn, tensor

from .activations import _is_glu
from .types import FeedForwardParams


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    This module applies a two-layer feed-forward network to each position
    independently. It supports various activation functions including GLU
    variants (e.g., SwiGLU, GeGLU) which use gated mechanisms for improved
    performance.
    """

    def __init__(self, **kwargs: Unpack[FeedForwardParams]):
        """
        Position-wise feed-forward network.

        This module implements a two-layer feed-forward network with an activation
        function in between. The network expands the model dimension by a factor of
        d_ff and then projects it back to the original dimension.

        The feed-forward mechanism is implemented as follows:
        1. Linear transformation: d_model -> d_model * d_ff
        2. Apply activation function (supports GLU variants)
        3. Apply dropout for regularization
        4. Linear transformation: d_model * d_ff -> d_model

        For GLU-based activations (e.g., SwiGLU, GeGLU), the first linear layer
        output is split into two parts: one goes through the activation function
        and the other serves as a gate, which are then element-wise multiplied.

        Args:
            kwargs: FeedForwardParams
                d_model: int - dimension of the model
                device: torch.device - device to use for computation
                batch_size: int - batch size
                activation: ActivationFunctions - activation function to use
                d_ff: int - expansion factor for the hidden dimension
                dropout: float - dropout rate for regularization
        """
        super(FeedForward, self).__init__()
        self.kwargs = kwargs

        # For GLU variants, w_1 needs to output 2x for the gate
        multiplier = 2 if _is_glu(kwargs["activation"]) else 1
        self.w_1 = nn.Linear(
            kwargs["d_model"], kwargs["d_model"] * kwargs["d_ff"] * multiplier
        )

        self.activation_fn = kwargs["activation"].value()
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.w_2 = nn.Linear(kwargs["d_model"] * kwargs["d_ff"], kwargs["d_model"])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the feed-forward network.

        Applies the two-layer feed-forward transformation with activation and dropout.
        Handles both standard activations and GLU-based activations:
        - For standard activations: x -> w_1 -> activation -> dropout -> w_2
        - For GLU activations: x -> w_1 (outputs 2x) -> GLU (chunks & gates) -> dropout -> w_2

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
                Input tensor to be transformed

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
                Output tensor after feed-forward transformation
        """
        # w_1 projection (outputs 2x features for GLU variants)
        x = self.w_1(x)

        # Apply activation (GLU variants chunk internally)
        x = self.activation_fn(x)

        # Dropout and final projection
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class ReZero(nn.Module):
    """
    ReZero normalization layer.

    ReZero is a simple yet effective normalization technique that replaces
    traditional layer normalization. It introduces a learnable scalar parameter
    that is multiplied with the output of a residual connection, allowing the
    network to dynamically control the contribution of each layer.

    This approach has been shown to:
    - Enable training of deeper networks without normalization layers
    - Provide faster convergence in some cases
    - Reduce the number of parameters compared to LayerNorm
    - Allow the network to learn which layers to emphasize

    The ReZero transformation is: output = g * fn(x) where g starts near zero
    and is learned during training.

    Reference:
        "ReZero is All You Need: Fast Convergence at Large Depth"
        Bachlechner et al., 2020
    """

    def __init__(self, fn) -> None:
        """
        Initialize the ReZero wrapper.

        Args:
            fn: nn.Module or callable
                The function or module to wrap with ReZero normalization.
                This is typically an attention or feed-forward layer.
        """
        super(ReZero, self).__init__()
        self.g = nn.Parameter(tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs) -> Tensor:
        """
        Apply ReZero normalization to the wrapped function.

        The forward pass computes g * fn(x, **kwargs), where g is a learnable
        scalar parameter initialized close to zero. This allows the network to
        gradually learn how much each layer should contribute to the final output.

        Args:
            x: Tensor
                Input tensor to be processed by the wrapped function
            **kwargs: dict
                Additional keyword arguments to pass to the wrapped function
                (e.g., attention masks, dropout parameters)

        Returns:
            Tensor: The output scaled by the learned parameter g
        """
        return self.g * self.fn(x, **kwargs)
