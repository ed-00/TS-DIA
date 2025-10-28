#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Linear attention mechanisms and kernel functions.

This module implements efficient linear attention mechanisms that reduce the
O(n²) complexity of standard softmax attention to O(n). It includes:
1. Kernel functions for approximating softmax attention
2. Non-causal linear attention for bidirectional contexts
3. Causal linear attention for autoregressive tasks
4. Random feature generation for kernel approximation

The implementation is based on the Performer architecture, which uses random
Fourier features to approximate the softmax kernel.

References:
    Paper: "Rethinking Attention with Performers" (Choromanski et al., 2020)
           https://arxiv.org/abs/2009.14794

    Implementation: performer-pytorch by lucidrains
                   https://github.com/lucidrains/performer-pytorch/blob/fc8b78441b1e27eb5d9b01fc738a8772cee07127/performer_pytorch/performer_pytorch.py#L221
"""

import math
from distutils.version import LooseVersion

import torch
from einops import repeat
from torch import Tensor

# Check PyTorch version for QR decomposition compatibility
TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion("1.8.0")


def orthogonal_matrix_chunk(cols: int, device: torch.device | None = None) -> Tensor:
    """
    Generate an orthogonal matrix chunk using QR decomposition.

    Creates a random orthogonal matrix by performing QR decomposition on a
    random unstructured matrix. This ensures the resulting matrix has
    orthonormal columns, which is important for maintaining the approximation
    quality of the kernel.

    Args:
        cols: int
            Number of columns (and rows) for the square orthogonal matrix
        device: torch.device | None
            Device to create the matrix on. If None, uses CPU

    Returns:
        Tensor: Orthogonal matrix of shape (cols, cols) with orthonormal columns
    """
    unstructured_block = torch.randn((cols, cols), device=device)

    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    else:
        q, r = torch.qr(unstructured_block.cpu(), some=True)

    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(
    nb_rows: int,
    nb_columns: int,
    scaling: int = 0,
    device: torch.device | None = None,
) -> Tensor:
    """
    Generate a Gaussian orthogonal random matrix for kernel approximation.

    Creates a random projection matrix using orthogonal Gaussian features.
    This matrix is used to project queries and keys into a random feature space
    where the dot product approximates the softmax kernel.

    Args:
        nb_rows: int
            Number of rows in the output matrix (number of random features)
        nb_columns: int
            Number of columns in the output matrix (dimension of input vectors)
        scaling: int
            Scaling strategy for the matrix:
            - 0: Use random Gaussian scaling
            - 1: Use fixed scaling of sqrt(nb_columns)
        device: torch.device | None
            Device to create the matrix on

    Returns:
        Tensor: Random projection matrix of shape (nb_rows, nb_columns)

    Raises:
        ValueError: If scaling is not 0 or 1
    """
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    # Generate full orthogonal blocks
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    # Handle remaining rows
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    # Apply scaling
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones(
            (nb_rows,), device=device
        )
    else:
        raise ValueError(f"Invalid scaling {scaling}")

    return torch.diag(multiplier) @ final_matrix


def softmax_kernel(
    data: Tensor,
    projection_matrix: Tensor,
    is_query: bool,
    normalize_data: bool = True,
    eps: float = 1e-4,
) -> Tensor:
    """
    Apply softmax kernel transformation using random features.

    Transforms input data using random Fourier features to approximate the
    softmax kernel. This allows for efficient linear attention computation.

    The transformation uses the identity:
    softmax(QK^T) ≈ φ(Q)φ(K)^T
    where φ is the feature map implemented by this function.

    Args:
        data: Tensor
            Input tensor of shape (batch, heads, seq_len, dim)
        projection_matrix: Tensor
            Random projection matrix of shape (nb_features, dim)
        is_query: bool
            Whether the input is a query (True) or key (False).
            Affects normalization strategy
        normalize_data: bool
            Whether to normalize input data by 1/sqrt(sqrt(dim))
        eps: float
            Small constant for numerical stability

    Returns:
        Tensor: Transformed tensor of shape (batch, heads, seq_len, nb_features)
    """
    b, h, *_ = data.shape

    # Normalize data if requested
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5

    # Expand projection matrix for batch and head dimensions
    projection = repeat(projection_matrix, "j d -> b h j d", b=b, h=h)
    projection = projection.type_as(data)

    # Project data: (b, h, n, d) @ (b, h, d, j) -> (b, h, n, j)
    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), projection)

    # Compute diagonal correction term
    diag_data = data**2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer**2)
    diag_data = diag_data.unsqueeze(dim=-1)

    # Apply exponential with normalization
    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.amax(data_dash, dim=-1, keepdim=True).detach()
            )
            + eps
        )
    else:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()
            )
            + eps
        )

    return data_dash.type_as(data)


def linear_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """
    Compute non-causal linear attention.

    Implements efficient linear attention using the associative property of
    matrix multiplication. Instead of computing (QK^T)V with O(n²) complexity,
    it computes Q(K^TV) with O(n) complexity.

    This is the core operation for bidirectional linear attention.

    Args:
        q: Tensor
            Query tensor of shape (batch, heads, seq_len, features)
        k: Tensor
            Key tensor of shape (batch, heads, seq_len, features)
        v: Tensor
            Value tensor of shape (batch, heads, seq_len, dim_v)

    Returns:
        Tensor: Attention output of shape (batch, heads, seq_len, dim_v)
    """
    # Sum over sequence dimension: (b, h, features)
    k_cumsum = k.sum(dim=-2)

    # Compute normalization: (b, h, seq_len)
    D_inv = 1.0 / torch.einsum("...nd,...d->...n", q, k_cumsum.type_as(q))

    # Compute context: (b, h, features, dim_v)
    context = torch.einsum("...nd,...ne->...de", k, v)

    # Apply attention: (b, h, seq_len, dim_v)
    out = torch.einsum("...de,...nd,...n->...ne", context, q, D_inv)

    return out


def causal_linear_attention(
    q: Tensor, k: Tensor, v: Tensor, chunk_size: int = 128, eps: float = 1e-6
) -> Tensor:
    """
    Compute causal linear attention for autoregressive tasks.

    Implements causal (unidirectional) linear attention where each position can
    only attend to previous positions. This is done by maintaining cumulative
    sums and processing the sequence in chunks.

    This version does not require CUDA kernels and works on any device.

    Args:
        q: Tensor
            Query tensor of shape (batch, heads, seq_len, features)
        k: Tensor
            Key tensor of shape (batch, heads, seq_len, features)
        v: Tensor
            Value tensor of shape (batch, heads, seq_len, dim_v)
        chunk_size: int
            Size of chunks for processing the sequence. Larger chunks use more
            memory but may be faster
        eps: float
            Small constant for numerical stability in normalization

    Returns:
        Tensor: Causal attention output of shape (batch, heads, seq_len, dim_v)
    """
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    # Process sequence in chunks to maintain causality
    for q_chunk, k_chunk, v_chunk in zip(
        *map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))
    ):
        # Cumulative sum of keys
        k_cumsum = last_k_cumsum + k_chunk.cumsum(dim=-2)

        # Compute normalization with causality
        D_inv = 1.0 / (
            torch.einsum("...nd,...nd->...n", q_chunk, k_cumsum.type_as(q_chunk)) + eps
        )

        # Compute context with causality
        context = torch.einsum("...nd,...ne->...nde", k_chunk, v_chunk)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)

        # Apply attention
        out = torch.einsum("...nde,...nd,...n->...ne", context_cumsum, q_chunk, D_inv)

        # Update cumulative states
        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim=-2)
