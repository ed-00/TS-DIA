#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Multi-head attention module.

This module applies multi-head attention to the input tensor.
It supports both standard softmax attention and efficient linear attention variants.
The attention mechanism projects the input into multiple heads, computes attention scores,
and then combines the attention outputs.

Linear attention implementation is based on the Performer architecture:
    Paper: "Rethinking Attention with Performers" (Choromanski et al., 2020)
           https://arxiv.org/abs/2009.14794

    Reference implementation: performer-pytorch by lucidrains
           https://github.com/lucidrains/performer-pytorch/blob/fc8b78441b1e27eb5d9b01fc738a8772cee07127/performer_pytorch/performer_pytorch.py#L221
"""

import math
from typing_extensions import Unpack

import torch
from torch import Tensor
from torch.nn import Dropout, Linear, Module
from torch.nn import functional as F

from .linear_attention import (
    causal_linear_attention,
    gaussian_orthogonal_random_matrix,
    linear_attention,
    softmax_kernel,
)
from .types import MultiHeadAttentionParams


class MultiHeadAttention(Module):
    """
    Multi-head attention module.

    This module applies multi-head attention to the input tensor.
    It projects the input tensor into multiple heads, computes the attention scores,
    and then combines the attention outputs.
    """

    def __init__(self, **kwargs: Unpack[MultiHeadAttentionParams]):
        """
        Multi-head attention module.

        This module applies multi-head attention to the input tensor.
        It supports both standard softmax attention (O(n²)) and efficient linear
        attention variants (O(n)) using kernel approximations.

        The attention mechanism is implemented as follows:
        1. Project the input tensor into multiple heads.
        2. Compute the attention scores:
           - Softmax: Standard scaled dot-product attention with softmax
           - Linear: Kernel-based linear attention using random features
           - Causal Linear: Causal linear attention for autoregressive tasks
        3. Compute the weighted sum of the value vectors.
        4. Project the combined attention outputs back to the original embedding dimension.

        The module supports masking to prevent the attention mechanism from attending to future tokens.

        Args:
            kwargs: MultiHeadAttentionParams
                device: torch.device
                d_model: int
                num_heads: int
                dropout: float
                batch_size: int
                attention_type: AttentionType - "softmax", "linear", or "causal_linear"
                nb_features: int | None - number of random features for linear attention
        """
        super(MultiHeadAttention, self).__init__()
        self.device = kwargs["device"]

        self.d_model = kwargs["d_model"]
        self.num_heads = kwargs["num_heads"]
        assert self.d_model % self.num_heads == 0, (
            "d_model must be divisible by num_heads"
        )
        self.d_head = self.d_model // self.num_heads

        self.dropout = kwargs["dropout"]
        self.batch_size = kwargs["batch_size"]

        # Attention type configuration
        self.attention_type = kwargs.get("attention_type", "softmax")

        # Per-attention-mode extra params
        # For causal linear attention we process the sequence in chunks; the
        # chunk size can be tuned to trade memory for speed. Use a safe default
        # of 128 if the caller provides None or invalid values.
        causal_chunk_size = kwargs.get("causal_chunk_size", 128)
        if not isinstance(causal_chunk_size, int) or causal_chunk_size <= 0:
            causal_chunk_size = 128
        self.causal_chunk_size = causal_chunk_size

        # Linear attention setup
        if self.attention_type in ("linear", "causal_linear"):
            # Number of random features for kernel approximation
            self.nb_features = kwargs.get("nb_features") or int(
                self.d_head * math.log(self.d_head)
            )

            # Create projection matrix for kernel approximation
            projection_matrix = gaussian_orthogonal_random_matrix(
                nb_rows=self.nb_features,
                nb_columns=self.d_head,
                scaling=0,
                device=self.device,
            )
            self.register_buffer("projection_matrix", projection_matrix)
        else:
            self.nb_features = None
            self.projection_matrix = None

        # Projection matrices for Q, K, V
        self.w_q = Linear(self.d_model, self.d_model, device=self.device)
        self.w_k = Linear(self.d_model, self.d_model, device=self.device)
        self.w_v = Linear(self.d_model, self.d_model, device=self.device)

        self.linear_out = Linear(
            self.d_model, self.d_model, device=self.device)

        self.dropout_attn = Dropout(self.dropout)
        self.dropout_out = Dropout(self.dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass of multi-head attention.

        Computes attention using either standard softmax attention or efficient
        linear attention based on the attention_type specified during initialization.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
                Input tensor to apply attention to
            mask: Tensor of shape (batch_size, seq_len, seq_len) | None
                Attention mask. For softmax attention, positions with mask=0 are masked.
                For linear attention, mask is applied to values before attention.

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
                Output tensor after applying multi-head attention
        """
        batch_size = x.shape[0]

        # Project to Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape to (batch, seq_len, num_heads, d_head)
        q = q.reshape(batch_size, -1, self.num_heads, self.d_head)
        k = k.reshape(batch_size, -1, self.num_heads, self.d_head)
        v = v.reshape(batch_size, -1, self.num_heads, self.d_head)

        # Transpose to (batch, num_heads, seq_len, d_head) for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention based on type
        if self.attention_type == "softmax":
            # Standard softmax attention
            attn_weights = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

            if mask is not None:
                # Expand mask for multi-head: (B, seq, seq) -> (B, 1, seq, seq)
                expanded_mask = mask.unsqueeze(1)
                attn_weights = attn_weights.masked_fill(
                    expanded_mask == 0, float("-inf")
                )

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout_attn(attn_weights)

            attn_output = attn_weights @ v

        elif self.attention_type == "linear":
            # Linear attention with kernel approximation
            if mask is not None:
                # Apply mask to values: expand (B, seq, seq) -> (B, 1, seq, 1)
                # We use the diagonal or last column of the mask to determine which positions are valid
                global_mask = mask[:, 0, :].unsqueeze(
                    1).unsqueeze(-1)  # (B, 1, seq, 1)
                v = v.masked_fill(global_mask == 0, 0.0)

            # Transform Q and K using softmax kernel
            assert self.projection_matrix is not None, "projection_matrix should not be None for linear attention"
            q_prime = softmax_kernel(
                q, self.projection_matrix, is_query=True, normalize_data=True
            )
            k_prime = softmax_kernel(
                k, self.projection_matrix, is_query=False, normalize_data=True
            )

            # Compute linear attention
            attn_output = linear_attention(q_prime, k_prime, v)

        elif self.attention_type == "causal_linear":
            # Causal linear attention for autoregressive tasks
            if mask is not None:
                # Apply mask to values: expand (B, seq, seq) -> (B, 1, seq, 1)
                global_mask = mask[:, 0, :].unsqueeze(
                    1).unsqueeze(-1)  # (B, 1, seq, 1)
                v = v.masked_fill(global_mask == 0, 0.0)

            # Transform Q and K using softmax kernel
            assert self.projection_matrix is not None, "projection_matrix should not be None for causal_linear attention"
            q_prime = softmax_kernel(
                q, self.projection_matrix, is_query=True, normalize_data=True
            )
            k_prime = softmax_kernel(
                k, self.projection_matrix, is_query=False, normalize_data=True
            )

            # Compute causal linear attention
            # Pass configured chunk size into the causal implementation so
            # callers (model / config) can tune memory usage at runtime.
            attn_output = causal_linear_attention(
                q_prime, k_prime, v, chunk_size=self.causal_chunk_size
            )
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        # Transpose back to (batch, seq_len, num_heads, d_head)
        attn_output = attn_output.transpose(1, 2)

        # Reshape to (batch, seq_len, d_model)
        attn_output = attn_output.reshape(batch_size, -1, self.d_model)

        # Final projection and dropout
        attn_output = self.linear_out(attn_output)
        attn_output = self.dropout_out(attn_output)

        return attn_output


class CrossAttention(MultiHeadAttention):
    """
    Multi-head cross-attention module.

    This module applies multi-head cross-attention to the input tensor.
    It projects the input tensor into multiple heads, computes the attention scores,
    and then combines the attention outputs.
    """

    def __init__(self, **kwargs: Unpack[MultiHeadAttentionParams]):
        """
        Multi-head cross-attention module.

        This module applies multi-head cross-attention to the input tensor.
        It projects the input tensor into multiple heads, computes the attention scores,
        and then combines the attention outputs.

        The attention mechanism is implemented as follows:
        1. Project the input tensor into multiple heads.
        2. Compute the attention scores between the query and key tensors.
        3. Apply softmax normalization to obtain attention weights.
        4. Compute the weighted sum of the value vectors.
        5. Project the combined attention outputs back to the original embedding dimension.

        The module supports masking to prevent the attention mechanism from attending to future tokens.

        Args:
            kwargs: MultiHeadAttentionParams
                device: torch.device
                d_model: int
                num_heads: int
                dropout: float
                batch_size: int
                attention_type: AttentionType - defaults to "softmax"
                nb_features: int | None - for linear attention
        """
        if "attention_type" not in kwargs:
            kwargs["attention_type"] = "softmax"
        if "nb_features" not in kwargs:
            kwargs["nb_features"] = None

        super(CrossAttention, self).__init__(**kwargs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Applies multi-head cross-attention between the query and key/value tensors.

        This method computes the attention scores between the query tensor `x` and the key/value tensor `mask`,
        applies softmax normalization to obtain attention weights, and then computes the weighted sum of the value vectors.
        The result is projected back to the original embedding dimension.

        Note: The parameter is named 'mask' for compatibility with the parent class, but it serves as the 
        key/value tensor for cross-attention when provided.

        Args:
            x (torch.Tensor): Query tensor of shape (batch_size, N_q, D).
            mask (torch.Tensor | None): Key/Value tensor of shape (batch_size, N_kv, D), typically from the encoder embeddings.
                                       If None, performs self-attention using x for both query and key/value.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, N_q, D) after applying cross-attention.
        """
        # Use x as both query and key/value if mask is None (self-attention)
        q = x
        kv = mask if mask is not None else x

        # q: (B, N_q, D) - Query
        # kv: (B, N_kv, D) - Key/Value from embedding encoder
        batch_size = q.shape[0]

        q_h = (
            self.w_q(q)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        k_h = (
            self.w_k(kv)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        v_h = (
            self.w_v(kv)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_head, dtype=torch.float32)
        )

        attention_weights = F.softmax(scores, dim=-1)
        p_att = self.dropout_attn(attention_weights)

        context_out = torch.matmul(p_att, v_h)
        context_out = (
            context_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_head)
        )

        return self.linear_out(context_out)
