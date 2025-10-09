#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Multi-head attention module.

This module applies multi-head attention to the input tensor.
It projects the input tensor into multiple heads, computes the attention scores,
and then combines the attention outputs.
"""
import torch
from torch import Tensor
from typing import Unpack
from torch.nn import functional as F
from .types import MultiHeadAttentionParams
from torch.nn import Dropout, Linear, Module



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

        ## share parameter for WQ,WK,WV
        self.w_q = Linear(self.d_model, self.d_model, device=self.device)
        self.w_k = Linear(self.d_model, self.d_model, device=self.device)
        self.w_v = Linear(self.d_model, self.d_model, device=self.device)

        self.linear_out = Linear(self.d_model, self.d_model, device=self.device)

        self.dropout_attn = Dropout(self.dropout)
        self.dropout_out = Dropout(self.dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            mask: Tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.reshape(self.batch_size, -1, self.num_heads, self.d_head)
        k = k.reshape(self.batch_size, -1, self.num_heads, self.d_head)
        v = v.reshape(self.batch_size, -1, self.num_heads, self.d_head)

        attn_weights = q @ k.transpose(-2, -1) / torch.sqrt(self.d_head)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        attn_output = attn_weights @ v
        attn_output = attn_output.reshape(self.batch_size, -1, self.d_model)
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
        """
        super(CrossAttention, self).__init__()

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Applies multi-head cross-attention between the query and key/value tensors.

        This method computes the attention scores between the query tensor `q` and the key/value tensor `kv`,
        applies softmax normalization to obtain attention weights, and then computes the weighted sum of the value vectors.
        The result is projected back to the original embedding dimension.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, N_q, D), typically from the attractor decoder.
            kv (torch.Tensor): Key/Value tensor of shape (batch_size, N_kv, D), typically from the encoder embeddings.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, N_q, D) after applying cross-attention.
        """
        # q: (B, N_q, D) - Query from attractor decoder
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

        context = torch.matmul(p_att, v_h)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_head)
        )

        return self.linear_out(context)
