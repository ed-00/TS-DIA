#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Performer transformer architecture.

This module implements the complete Performer transformer with efficient
linear attention. It provides:
1. PerformerLayer: Single transformer layer with attention + feed-forward
2. Performer: Complete encoder/decoder stack with multiple layers
3. Support for various configurations (ReZero, PreScaleNorm, etc.)

The Performer uses FAVOR+ algorithm for efficient attention with O(n) complexity
instead of O(n²) in standard transformers.

References:
    Paper: "Rethinking Attention with Performers" (Choromanski et al., 2020)
           https://arxiv.org/abs/2009.14794

    Implementation: performer-pytorch by lucidrains
                   https://github.com/lucidrains/performer-pytorch/blob/fc8b78441b1e27eb5d9b01fc738a8772cee07127/performer_pytorch/performer_pytorch.py
"""

from typing import Unpack

import torch
from torch import Tensor, nn

from .activations import ActivationFunctions
from .attention import CrossAttention, MultiHeadAttention
from .feedforward import FeedForward, ReZero
from .norm import PreLayerNorm, PreScaleNorm
from .types import PerformerLayerParams, PerformerParams
from .utils import ProjectionUpdater


class PerformerLayer(nn.Module):
    """
    Universal Performer transformer layer.

    Can be used as:
    - **Encoder layer**: Self-attention + FFN
    - **Decoder layer**: Self-attention + Cross-attention + FFN

    Implements a transformer layer with:
    - Multi-head self-attention (standard or linear)
    - Optional cross-attention (for encoder-decoder)
    - Feed-forward network
    - Residual connections
    - Pre-normalization (LayerNorm, ScaleNorm, or ReZero)

    The layer follows the pre-norm architecture:
        x = x + self_attn(norm(x))
        x = x + cross_attn(norm(x), encoder_output)  # if decoder
        x = x + ffn(norm(x))
    """

    def __init__(self, **kwargs: Unpack[PerformerLayerParams]):
        """
        Initialize universal Performer layer.

        Args:
            kwargs: PerformerLayerParams
                d_model: int - model dimension
                device: torch.device - computation device
                batch_size: int - batch size
                num_heads: int - number of attention heads
                d_ff: int - feed-forward expansion factor
                dropout: float - dropout rate
                attention_type: AttentionType - "softmax", "linear", or "causal_linear"
                activation: ActivationFunctions - activation for feed-forward
                use_cross_attention: bool - add cross-attention (for decoder)
                use_rezero: bool - use ReZero normalization
                use_scalenorm: bool - use PreScaleNorm normalization
                nb_features: int | None - random features for linear attention
        """
        super().__init__()

        d_model = kwargs["d_model"]
        device = kwargs["device"]
        batch_size = kwargs["batch_size"]
        num_heads = kwargs["num_heads"]
        d_ff = kwargs["d_ff"]
        dropout = kwargs["dropout"]
        attention_type = kwargs.get("attention_type", "softmax")
        activation = kwargs.get("activation", ActivationFunctions.GELU)
        use_cross_attention = kwargs.get("use_cross_attention", False)
        use_rezero = kwargs.get("use_rezero", False)
        use_scalenorm = kwargs.get("use_scalenorm", False)
        nb_features = kwargs.get("nb_features")

        self.use_cross_attention = use_cross_attention

        # Create self-attention layer
        self.self_attention = MultiHeadAttention(
            device=device,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_size=batch_size,
            attention_type=attention_type,
            nb_features=nb_features,
        )

        # Create cross-attention layer (for decoder)
        if use_cross_attention:
            self.cross_attention = CrossAttention(
                device=device,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_size=batch_size,
                attention_type="softmax",  # Cross-attention typically uses softmax
                nb_features=None,
            )

        # Create feed-forward layer
        self.feed_forward = FeedForward(
            d_model=d_model,
            device=device,
            batch_size=batch_size,
            activation=activation,
            d_ff=d_ff,
            dropout=dropout,
        )

        if use_rezero:
            # ReZero: learnable scalar, no normalization
            self.norm_self_attn = ReZero(self.self_attention)
            if use_cross_attention:
                # Store just the ReZero params for cross-attention
                self.norm_cross_attn = ReZero(nn.Identity())  # Dummy wrapper
            self.norm_ff = ReZero(self.feed_forward)
        elif use_scalenorm:
            # PreScaleNorm: L2 normalization with gain
            self.norm_self_attn = PreScaleNorm(
                dim=d_model, fn=self.self_attention, eps=1e-5
            )
            if use_cross_attention:
                # Store just the PreScaleNorm params for cross-attention
                self.norm_cross_attn = PreScaleNorm(
                    dim=d_model, fn=nn.Identity(), eps=1e-5
                )
            self.norm_ff = PreScaleNorm(dim=d_model, fn=self.feed_forward, eps=1e-5)
        else:
            # PreLayerNorm: standard layer normalization (default)
            self.norm_self_attn = PreLayerNorm(dim=d_model, fn=self.self_attention)
            if use_cross_attention:
                # Store just the LayerNorm for cross-attention
                self.norm_cross_attn = PreLayerNorm(dim=d_model, fn=nn.Identity())
            self.norm_ff = PreLayerNorm(dim=d_model, fn=self.feed_forward)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass of universal Performer layer.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
                Input tensor
            encoder_output: Tensor of shape (batch_size, src_len, d_model) | None
                Encoder output for cross-attention (required if use_cross_attention=True)
            self_attn_mask: Tensor of shape (batch_size, seq_len, seq_len) | None
                Self-attention mask
            cross_attn_mask: Tensor of shape (batch_size, seq_len, src_len) | None
                Cross-attention mask
            **kwargs: Additional arguments

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
                Output tensor after all sublayers
        """
        # Self-attention with residual connection
        x = x + self.norm_self_attn(x, mask=self_attn_mask)

        # Cross-attention with residual connection (for decoder)
        if self.use_cross_attention:
            if encoder_output is None:
                raise ValueError(
                    "encoder_output is required when use_cross_attention=True"
                )

            # Apply normalization before cross-attention
            if hasattr(self, "norm_cross_attn"):
                # For ReZero, just multiply by learnable scalar
                if isinstance(self.norm_cross_attn, ReZero):
                    cross_output = self.cross_attention(x, encoder_output)
                    x = x + cross_output * self.norm_cross_attn.g
                # For PreScaleNorm, normalize then apply
                elif isinstance(self.norm_cross_attn, PreScaleNorm):
                    n = torch.norm(x, dim=-1, keepdim=True).clamp(
                        min=self.norm_cross_attn.eps
                    )
                    x_normed = x / n * self.norm_cross_attn.g
                    x = x + self.cross_attention(x_normed, encoder_output)
                # For PreLayerNorm, normalize then apply
                elif isinstance(self.norm_cross_attn, PreLayerNorm):
                    x_normed = self.norm_cross_attn.norm(x)
                    x = x + self.cross_attention(x_normed, encoder_output)
                else:
                    x = x + self.cross_attention(x, encoder_output)
            else:
                x = x + self.cross_attention(x, encoder_output)

        # Feed-forward with residual connection
        x = x + self.norm_ff(x)

        return x


class Performer(nn.Module):
    """
    Complete Performer transformer (encoder or decoder).

    Stacks multiple PerformerLayer modules with optional projection matrix
    redrawing for linear attention variants. Supports various normalization
    strategies and can be used as an encoder or decoder.

    For linear attention, projection matrices can be periodically redrawn
    during training to maintain approximation quality.
    """

    def __init__(self, **kwargs: Unpack[PerformerParams]):
        """
        Initialize Performer transformer.

        Args:
            kwargs: PerformerParams
                d_model: int - model dimension
                device: torch.device - computation device
                batch_size: int - batch size
                num_layers: int - number of transformer layers
                num_heads: int - number of attention heads
                d_ff: int - feed-forward expansion factor
                dropout: float - dropout rate
                attention_type: AttentionType - attention mechanism type
                activation: ActivationFunctions - activation function
                use_rezero: bool - use ReZero normalization
                use_scalenorm: bool - use PreScaleNorm normalization
                nb_features: int | None - random features for linear attention
                feature_redraw_interval: int | None - redraw interval
                auto_check_redraw: bool - auto redraw projections
        """
        super().__init__()

        num_layers = kwargs["num_layers"]
        self.auto_check_redraw = kwargs.get("auto_check_redraw", True)
        feature_redraw_interval = kwargs.get("feature_redraw_interval", 1000)

        # Create transformer layers
        self.layers = nn.ModuleList(
            [
                PerformerLayer(
                    d_model=kwargs["d_model"],
                    device=kwargs["device"],
                    batch_size=kwargs["batch_size"],
                    num_heads=kwargs["num_heads"],
                    d_ff=kwargs["d_ff"],
                    dropout=kwargs["dropout"],
                    attention_type=kwargs.get("attention_type", "softmax"),
                    activation=kwargs.get("activation", ActivationFunctions.GELU),
                    use_cross_attention=kwargs.get("use_cross_attention", False),
                    use_rezero=kwargs.get("use_rezero", False),
                    use_scalenorm=kwargs.get("use_scalenorm", False),
                    nb_features=kwargs.get("nb_features"),
                )
                for _ in range(num_layers)
            ]
        )

        # Projection updater for linear attention
        self.proj_updater = ProjectionUpdater(self.layers, feature_redraw_interval)

    def fix_projection_matrices_(self) -> None:
        """
        Fix projection matrices permanently (disable redrawing).

        Call this after training to freeze projection matrices for inference.
        """
        self.proj_updater.fix_projections_()

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass through all Performer layers.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
                Input tensor
            encoder_output: Tensor of shape (batch_size, src_len, d_model) | None
                Encoder output for cross-attention (used by decoder layers)
            self_attn_mask: Tensor of shape (batch_size, seq_len, seq_len) | None
                Self-attention mask
            cross_attn_mask: Tensor of shape (batch_size, seq_len, src_len) | None
                Cross-attention mask
            **kwargs: Additional arguments

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
                Output after all transformer layers
        """
        # Optionally redraw projection matrices
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()

        # Pass through all layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_output=encoder_output,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                **kwargs,
            )

        return x


class PerformerEncoder(Performer):
    """
    Performer encoder with optional final normalization and output projection.

    Encoder-only architecture for bidirectional encoding of input sequences.
    Uses self-attention without cross-attention.
    Can include a task-specific output projection for classification tasks.
    """

    def __init__(self, **kwargs: Unpack[PerformerParams]):
        """
        Initialize Performer encoder.

        Args:
            kwargs: PerformerParams - see Performer for details
                Note: use_cross_attention will be forced to False
                num_classes: int | None - if provided, adds output projection layer
        """
        # Ensure encoder doesn't have cross-attention
        kwargs["use_cross_attention"] = False
        super().__init__(**kwargs)

        # Optional final layer normalization
        if not kwargs.get("use_rezero", False):
            self.final_norm = nn.LayerNorm(kwargs["d_model"])
        else:
            self.final_norm = None

        # Optional output projection for classification tasks
        num_classes = kwargs.get("num_classes")
        if num_classes is not None:
            self.output_proj = nn.Linear(kwargs["d_model"], num_classes)
        else:
            self.output_proj = None

    def forward(self, x: Tensor, mask: Tensor | None = None, **kwargs) -> Tensor:
        """
        Encode input sequence.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
                Input embeddings
            mask: Tensor of shape (batch_size, seq_len, seq_len) | None
                Self-attention mask
            **kwargs: Additional arguments

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) or
            Tensor of shape (batch_size, seq_len, num_classes) if output_proj exists
                Encoded representations
        """
        x = super().forward(x, self_attn_mask=mask, **kwargs)

        # Apply final normalization if using LayerNorm
        if self.final_norm is not None:
            x = self.final_norm(x)

        # Apply output projection if exists
        if self.output_proj is not None:
            x = self.output_proj(x)

        return x


class PerformerDecoder(Performer):
    """
    Performer decoder with cross-attention for encoder-decoder models.

    Decoder architecture with:
    - Causal self-attention
    - Cross-attention to encoder outputs
    - Feed-forward networks

    Can be used standalone for language modeling or with encoder for seq2seq tasks.
    """

    def __init__(self, **kwargs: Unpack[PerformerParams]):
        """
        Initialize Performer decoder.

        Args:
            kwargs: PerformerParams - see Performer for details
                use_cross_attention: bool - add cross-attention (default True for decoder)
                attention_type: AttentionType - typically "causal_linear" for self-attention
        """
        # Decoders typically have cross-attention for encoder-decoder models
        # Set to True by default but can be overridden for decoder-only models
        if "use_cross_attention" not in kwargs:
            kwargs["use_cross_attention"] = True

        super().__init__(**kwargs)

        # Optional final layer normalization
        if not kwargs.get("use_rezero", False):
            self.final_norm = nn.LayerNorm(kwargs["d_model"])
        else:
            self.final_norm = None

    def create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Create causal attention mask.

        Args:
            seq_len: int - sequence length
            device: torch.device - device for tensor

        Returns:
            Tensor of shape (1, seq_len, seq_len) - causal mask
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        use_causal_mask: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Decode sequence with optional encoder context.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
                Input embeddings
            encoder_output: Tensor of shape (batch_size, src_len, d_model) | None
                Encoder output for cross-attention
            self_attn_mask: Tensor of shape (batch_size, seq_len, seq_len) | None
                Custom self-attention mask
            cross_attn_mask: Tensor of shape (batch_size, seq_len, src_len) | None
                Cross-attention mask
            use_causal_mask: bool - automatically create causal mask for self-attention
            **kwargs: Additional arguments

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
                Decoded representations
        """
        # Create causal mask if requested and no mask provided
        if use_causal_mask and self_attn_mask is None:
            batch_size, seq_len, _ = x.shape
            self_attn_mask = self.create_causal_mask(seq_len, x.device)
            self_attn_mask = self_attn_mask.expand(batch_size, -1, -1)

        x = super().forward(
            x,
            encoder_output=encoder_output,
            self_attn_mask=self_attn_mask,
            cross_attn_mask=cross_attn_mask,
            **kwargs,
        )

        # Apply final normalization if using LayerNorm
        if self.final_norm is not None:
            x = self.final_norm(x)

        return x


class EncoderDecoderTransformer(nn.Module):
    """
    Complete encoder-decoder Performer transformer.

    Classic sequence-to-sequence architecture with:
    - Encoder: bidirectional self-attention on source sequence
    - Decoder: causal self-attention + cross-attention on target sequence

    Suitable for translation, summarization, and other seq2seq tasks.
    """

    def __init__(
        self, encoder_params: PerformerParams, decoder_params: PerformerParams
    ):
        """
        Initialize encoder-decoder transformer.

        Args:
            encoder_params: PerformerParams
                Configuration for encoder (use_cross_attention will be forced to False)
            decoder_params: PerformerParams
                Configuration for decoder (use_cross_attention will be forced to True)
        """
        super().__init__()

        # Create encoder (no cross-attention)
        self.encoder = PerformerEncoder(**encoder_params)

        # Create decoder (with cross-attention)
        self.decoder = PerformerDecoder(**decoder_params)

    def encode(self, src: Tensor, src_mask: Tensor | None = None, **kwargs) -> Tensor:
        """
        Encode source sequence.

        Args:
            src: Tensor of shape (batch_size, src_len, d_model)
                Source embeddings
            src_mask: Tensor of shape (batch_size, src_len, src_len) | None
                Source attention mask
            **kwargs: Additional arguments

        Returns:
            Tensor of shape (batch_size, src_len, d_model)
                Encoded source representations
        """
        return self.encoder(src, mask=src_mask, **kwargs)

    def decode(
        self,
        tgt: Tensor,
        encoder_output: Tensor,
        tgt_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
        use_causal_mask: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Decode target sequence with encoder context.

        Args:
            tgt: Tensor of shape (batch_size, tgt_len, d_model)
                Target embeddings
            encoder_output: Tensor of shape (batch_size, src_len, d_model)
                Encoded source representations
            tgt_mask: Tensor of shape (batch_size, tgt_len, tgt_len) | None
                Target self-attention mask (typically causal)
            src_mask: Tensor of shape (batch_size, tgt_len, src_len) | None
                Cross-attention mask (which source positions to attend to)
            use_causal_mask: bool - automatically create causal mask for target
            **kwargs: Additional arguments

        Returns:
            Tensor of shape (batch_size, tgt_len, d_model)
                Decoded target representations
        """
        return self.decoder(
            tgt,
            encoder_output=encoder_output,
            self_attn_mask=tgt_mask,
            cross_attn_mask=src_mask,
            use_causal_mask=use_causal_mask,
            **kwargs,
        )

    def forward(
        self,
        src: Tensor | None = None,
        tgt: Tensor | None = None,
        x: Tensor | None = None,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        use_causal_mask: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src: Tensor of shape (batch_size, src_len, d_model)
                Source embeddings (if None, uses x)
            tgt: Tensor of shape (batch_size, tgt_len, d_model)
                Target embeddings (if None, uses x)
            x: Tensor of shape (batch_size, seq_len, d_model)
                Unified input for both src and tgt (for diarization tasks)
            src_mask: Tensor of shape (batch_size, src_len, src_len) | None
                Source self-attention mask
            tgt_mask: Tensor of shape (batch_size, tgt_len, tgt_len) | None
                Target self-attention mask
            cross_attn_mask: Tensor of shape (batch_size, tgt_len, src_len) | None
                Cross-attention mask
            use_causal_mask: bool - automatically create causal mask for target
            **kwargs: Additional arguments

        Returns:
            Tensor of shape (batch_size, tgt_len, d_model)
                Decoder output (ready for output projection)
        """
        # Handle unified input (for diarization where src and tgt are the same)
        if x is not None:
            src = x if src is None else src
            tgt = x if tgt is None else tgt

        if src is None or tgt is None:
            raise ValueError("Must provide either (src and tgt) or x")

        # Encode source
        encoder_output = self.encode(src, src_mask=src_mask, **kwargs)

        # Decode target with encoder context
        decoder_output = self.decode(
            tgt,
            encoder_output=encoder_output,
            tgt_mask=tgt_mask,
            src_mask=cross_attn_mask,
            use_causal_mask=use_causal_mask,
            **kwargs,
        )

        return decoder_output
