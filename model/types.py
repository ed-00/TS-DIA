from dataclasses import dataclass
from typing import TypedDict
from torch import device


@dataclass
class TransformerSharedParams(TypedDict):
    """Shared parameters for all transformer modules"""
    d_model: int # dimension of the model
    device: device # device to use
    batch_size: int # batch size to use


@dataclass
class MultiHeadAttentionParams(TransformerSharedParams):
    num_heads: int # number of heads
    dropout: float # dropout rate for the attention weights
