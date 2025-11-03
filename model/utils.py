#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Utility functions and helpers for transformer models.

This module provides common utility functions used across transformer
implementations, including tensor manipulation, module inspection, and
projection matrix management for linear attention.

References:
    Based on utilities from Performer architecture
    https://github.com/lucidrains/performer-pytorch/blob/fc8b78441b1e27eb5d9b01fc738a8772cee07127/performer_pytorch/performer_pytorch.py
"""

from typing import Any, Type

import torch
from torch import Tensor, nn


def exists(val: Any) -> bool:
    """
    Check if a value is not None.

    Args:
        val: Any value to check

    Returns:
        bool: True if val is not None, False otherwise
    """
    return val is not None


def default(val: Any, default_val: Any) -> Any:
    """
    Return val if it exists, otherwise return default_val.

    Args:
        val: Primary value to return if it exists
        default_val: Default value to return if val is None

    Returns:
        val if val is not None, else default_val
    """
    return val if exists(val) else default_val


def cast_tuple(val: Any) -> tuple:
    """
    Convert value to tuple if it isn't already.

    Args:
        val: Value to convert to tuple

    Returns:
        tuple: val wrapped in tuple if not already a tuple, otherwise val itself
    """
    return (val,) if not isinstance(val, tuple) else val


def get_module_device(module: nn.Module) -> torch.device:
    """
    Get the device of a PyTorch module.

    Args:
        module: PyTorch module to inspect

    Returns:
        torch.device: Device where the module's parameters are stored
    """
    return next(module.parameters()).device


def find_modules(nn_module: nn.Module, module_type: Type[nn.Module]) -> list[nn.Module]:
    """
    Find all modules of a specific type within a neural network.

    Recursively searches through a module and all its children to find
    instances of a specific module type.

    Args:
        nn_module: Root module to search
        module_type: Type of module to find (e.g., nn.Linear)

    Returns:
        list[nn.Module]: List of all modules of the specified type
    """
    return [module for module in nn_module.modules() if isinstance(module, module_type)]


class ProjectionUpdater(nn.Module):
    """
    Manages periodic redrawing of random projection matrices for linear attention.

    This module tracks training iterations and triggers redrawing of projection
    matrices at specified intervals. This is important for maintaining good
    approximation quality during training with linear attention.

    The random features used in linear attention can become stale during training,
    so periodically redrawing them helps maintain performance.

    Reference:
        Performer paper section on "Redrawing Random Features"
        https://arxiv.org/abs/2009.14794
    """

    def __init__(self, instance: nn.Module, feature_redraw_interval: int | None):
        """
        Initialize projection updater.

        Args:
            instance: Module containing attention layers with projection matrices
            feature_redraw_interval: Number of forward passes between redraws.
                If None, projections are never redrawn (fixed after initialization)
        """
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer("calls_since_last_redraw", torch.tensor(0))

    def fix_projections_(self) -> None:
        """
        Fix projection matrices permanently (disable redrawing).

        Call this after training to freeze projection matrices for inference.
        """
        self.feature_redraw_interval = None

    def redraw_projections(self) -> None:
        """
        Redraw projection matrices if interval has been reached.

        This method:
        1. Checks if we're in training mode
        2. Checks if redraw interval has been reached
        3. Finds all MultiHeadAttention modules with linear attention
        4. Redraws their projection matrices
        5. Resets the counter
        """
        if not self.training:
            return

        if self.feature_redraw_interval is None:
            return

        if self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self.instance)

            # Import here to avoid circular dependency
            from .attention import MultiHeadAttention

            # Find all attention modules
            attentions = find_modules(self.instance, MultiHeadAttention)

            # Redraw projection matrices for linear attention variants
            for attn in attentions:
                if (
                    hasattr(attn, "projection_matrix")
                    and attn.projection_matrix is not None
                    and hasattr(attn, "nb_features")
                    and hasattr(attn, "d_head")
                ):
                    # Redraw projection matrix
                    from .linear_attention import gaussian_orthogonal_random_matrix

                    # Get dimensions (these should be integers from the attention module)
                    nb_rows = attn.nb_features
                    nb_columns = attn.d_head

                    # Type guard to ensure we have integers
                    if isinstance(nb_rows, int) and isinstance(nb_columns, int):
                        new_projection = gaussian_orthogonal_random_matrix(
                            nb_rows=nb_rows,
                            nb_columns=nb_columns,
                            scaling=0,
                            device=device,
                        )
                        # Replace the projection matrix - ensure it's a tensor
                        if isinstance(attn.projection_matrix, Tensor):
                            with torch.no_grad():
                                attn.projection_matrix.copy_(new_projection)

            # Reset counter - ensure it's a tensor buffer
            if isinstance(self.calls_since_last_redraw, Tensor):
                with torch.no_grad():
                    self.calls_since_last_redraw.zero_()
        else:
            self.calls_since_last_redraw += 1

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass (not used directly, included for module compatibility).

        Args:
            x: Input tensor

        Returns:
            Input tensor unchanged

        Raises:
            NotImplementedError: This module is not meant to be called directly
        """
        raise NotImplementedError(
            "ProjectionUpdater should not be used in forward pass directly"
        )
