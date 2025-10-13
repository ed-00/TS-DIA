#!/usr/bin/env python
"""
Comprehensive test for Encoder-Decoder Performer transformer.

Tests:
- Universal PerformerLayer (encoder and decoder configurations)
- PerformerEncoder (encoder-only)
- PerformerDecoder (decoder with cross-attention)
- EncoderDecoderTransformer (full seq2seq)
- Different attention mechanisms and normalizations
"""

import torch

from model.activations import ActivationFunctions
from model.transformer import (
    EncoderDecoderTransformer,
    PerformerDecoder,
    PerformerEncoder,
    PerformerLayer,
)


def test_universal_layer():
    """Test universal PerformerLayer in encoder and decoder modes."""
    print("\n" + "=" * 70)
    print("Testing Universal PerformerLayer")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    seq_len = 32
    d_model = 256

    print("\n1. Testing as Encoder Layer (no cross-attention):")
    encoder_layer = PerformerLayer(
        d_model=d_model,
        device=device,
        batch_size=batch_size,
        num_heads=8,
        d_ff=4,
        dropout=0.1,
        attention_type="softmax",
        activation=ActivationFunctions.GELU,
        use_cross_attention=False,  # Encoder mode
    )

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    output = encoder_layer(x)

    assert output.shape == x.shape
    print(f"  ✓ Encoder layer output: {output.shape}")

    print("\n2. Testing as Decoder Layer (with cross-attention):")
    decoder_layer = PerformerLayer(
        d_model=d_model,
        device=device,
        batch_size=batch_size,
        num_heads=8,
        d_ff=4,
        dropout=0.1,
        attention_type="causal_linear",
        activation=ActivationFunctions.SWIGLU,
        use_cross_attention=True,  # Decoder mode
    )

    tgt = torch.randn(batch_size, seq_len, d_model, device=device)
    encoder_output = torch.randn(batch_size, 64, d_model, device=device)

    # Create causal mask
    causal_mask = torch.tril(torch.ones(batch_size, seq_len, seq_len, device=device))

    output = decoder_layer(
        tgt, encoder_output=encoder_output, self_attn_mask=causal_mask
    )

    assert output.shape == tgt.shape
    print(f"  ✓ Decoder layer output: {output.shape}")
    print(f"  ✓ Decoder has cross-attention: {decoder_layer.use_cross_attention}")

    print("\n✅ Universal PerformerLayer test passed!")


def test_encoder_only():
    """Test PerformerEncoder."""
    print("\n" + "=" * 70)
    print("Testing PerformerEncoder (Encoder-Only)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    seq_len = 64
    d_model = 512

    encoder = PerformerEncoder(
        d_model=d_model,
        device=device,
        batch_size=batch_size,
        num_layers=6,
        num_heads=8,
        d_ff=4,
        dropout=0.1,
        attention_type="linear",
        activation=ActivationFunctions.GEGLU,
        nb_features=256,
    ).to(device)

    src = torch.randn(batch_size, seq_len, d_model, device=device)
    encoded = encoder(src)

    print(f"  Source shape: {src.shape}")
    print(f"  Encoded shape: {encoded.shape}")
    assert encoded.shape == src.shape

    # Verify no cross-attention
    print(
        f"  ✓ Encoder has no cross-attention: {not encoder.layers[0].use_cross_attention}"
    )

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  ✓ Total parameters: {num_params:,}")

    print("\n✅ PerformerEncoder test passed!")


def test_decoder_with_cross_attention():
    """Test PerformerDecoder with cross-attention."""
    print("\n" + "=" * 70)
    print("Testing PerformerDecoder (Decoder with Cross-Attention)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    src_len = 64
    tgt_len = 32
    d_model = 512

    decoder = PerformerDecoder(
        d_model=d_model,
        device=device,
        batch_size=batch_size,
        num_layers=6,
        num_heads=8,
        d_ff=4,
        dropout=0.1,
        attention_type="causal_linear",
        activation=ActivationFunctions.SWIGLU,
        nb_features=256,
    ).to(device)

    tgt = torch.randn(batch_size, tgt_len, d_model, device=device)
    encoder_output = torch.randn(batch_size, src_len, d_model, device=device)

    # Decode with encoder context
    decoded = decoder(tgt, encoder_output=encoder_output, use_causal_mask=True)

    print(f"  Target shape: {tgt.shape}")
    print(f"  Encoder output shape: {encoder_output.shape}")
    print(f"  Decoded shape: {decoded.shape}")
    assert decoded.shape == tgt.shape

    # Verify cross-attention exists
    print(f"  ✓ Decoder has cross-attention: {decoder.layers[0].use_cross_attention}")

    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"  ✓ Total parameters: {num_params:,}")

    print("\n✅ PerformerDecoder test passed!")


def test_encoder_decoder_transformer():
    """Test complete EncoderDecoderTransformer."""
    print("\n" + "=" * 70)
    print("Testing EncoderDecoderTransformer (Full Seq2Seq)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    src_len = 64
    tgt_len = 48
    d_model = 512

    # Create encoder-decoder transformer
    model = EncoderDecoderTransformer(
        encoder_params={
            "d_model": d_model,
            "device": device,
            "batch_size": batch_size,
            "num_layers": 6,
            "num_heads": 8,
            "d_ff": 4,
            "dropout": 0.1,
            "attention_type": "linear",
            "activation": ActivationFunctions.GEGLU,
            "nb_features": 256,
        },
        decoder_params={
            "d_model": d_model,
            "device": device,
            "batch_size": batch_size,
            "num_layers": 6,
            "num_heads": 8,
            "d_ff": 4,
            "dropout": 0.1,
            "attention_type": "causal_linear",
            "activation": ActivationFunctions.SWIGLU,
            "nb_features": 256,
        },
    ).to(device)

    src = torch.randn(batch_size, src_len, d_model, device=device)
    tgt = torch.randn(batch_size, tgt_len, d_model, device=device)

    # Full forward pass
    output = model(src, tgt, use_causal_mask=True)

    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == tgt.shape

    # Test separate encode/decode
    print("\n  Testing separate encode/decode:")
    encoded = model.encode(src)
    print(f"  Encoded shape: {encoded.shape}")

    decoded = model.decode(tgt, encoded, use_causal_mask=True)
    print(f"  Decoded shape: {decoded.shape}")

    # Count total parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  ✓ Total parameters: {num_params:,}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print("  ✓ Gradient flow successful")

    print("\n✅ EncoderDecoderTransformer test passed!")


def test_different_configurations():
    """Test various configurations."""
    print("\n" + "=" * 70)
    print("Testing Different Configurations")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 32
    d_model = 256

    configs = [
        (
            "LayerNorm + Softmax",
            {"use_rezero": False, "use_scalenorm": False, "attention_type": "softmax"},
        ),
        (
            "ReZero + Linear",
            {"use_rezero": True, "attention_type": "linear", "nb_features": 128},
        ),
        (
            "ScaleNorm + Causal Linear",
            {
                "use_scalenorm": True,
                "attention_type": "causal_linear",
                "nb_features": 128,
            },
        ),
    ]

    for name, config in configs:
        print(f"\n  Testing: {name}")

        model = EncoderDecoderTransformer(
            encoder_params={
                "d_model": d_model,
                "device": device,
                "batch_size": batch_size,
                "num_layers": 2,
                "num_heads": 4,
                "d_ff": 4,
                "dropout": 0.1,
                "activation": ActivationFunctions.GELU,
                **config,
            },
            decoder_params={
                "d_model": d_model,
                "device": device,
                "batch_size": batch_size,
                "num_layers": 2,
                "num_heads": 4,
                "d_ff": 4,
                "dropout": 0.1,
                "activation": ActivationFunctions.SWIGLU,
                **config,
            },
        ).to(device)

        src = torch.randn(batch_size, seq_len, d_model, device=device)
        tgt = torch.randn(batch_size, seq_len, d_model, device=device)

        output = model(src, tgt)
        assert output.shape == tgt.shape
        print(f"    ✓ {name}: {output.shape}")

    print("\n✅ Configuration test passed!")


def test_masking():
    """Test different masking scenarios."""
    print("\n" + "=" * 70)
    print("Testing Masking Scenarios")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    src_len = 32
    tgt_len = 24
    d_model = 256

    model = EncoderDecoderTransformer(
        encoder_params={
            "d_model": d_model,
            "device": device,
            "batch_size": batch_size,
            "num_layers": 2,
            "num_heads": 4,
            "d_ff": 4,
            "dropout": 0.1,
        },
        decoder_params={
            "d_model": d_model,
            "device": device,
            "batch_size": batch_size,
            "num_layers": 2,
            "num_heads": 4,
            "d_ff": 4,
            "dropout": 0.1,
        },
    ).to(device)

    src = torch.randn(batch_size, src_len, d_model, device=device)
    tgt = torch.randn(batch_size, tgt_len, d_model, device=device)

    # Test 1: No masking
    print("\n  1. No masking:")
    output = model(src, tgt, use_causal_mask=False)
    print(f"    ✓ Output shape: {output.shape}")

    # Test 2: Causal masking (default)
    print("\n  2. Causal masking (autoregressive):")
    output = model(src, tgt, use_causal_mask=True)
    print(f"    ✓ Output shape: {output.shape}")

    # Test 3: Custom source mask (padding)
    print("\n  3. Custom source padding mask:")
    src_mask = torch.ones(batch_size, src_len, src_len, device=device)
    src_mask[:, :, -10:] = 0  # Mask last 10 positions
    output = model(src, tgt, src_mask=src_mask, use_causal_mask=True)
    print(f"    ✓ Output shape: {output.shape}")

    # Test 4: Custom cross-attention mask
    print("\n  4. Custom cross-attention mask:")
    cross_mask = torch.ones(batch_size, tgt_len, src_len, device=device)
    cross_mask[:, :, -10:] = 0  # Don't attend to last 10 source positions
    output = model(src, tgt, cross_attn_mask=cross_mask, use_causal_mask=True)
    print(f"    ✓ Output shape: {output.shape}")

    print("\n✅ Masking test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Encoder-Decoder Transformer Test Suite")
    print("=" * 70)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    test_universal_layer()
    test_encoder_only()
    test_decoder_with_cross_attention()
    test_encoder_decoder_transformer()
    test_different_configurations()
    test_masking()

    print("\n" + "=" * 70)
    print("✅ All Encoder-Decoder tests passed successfully!")
    print("=" * 70)

    print("\n🎉 Your complete Encoder-Decoder implementation is ready!")
    print("\nYou now have a universal transformer with:")
    print("\n  📦 Components:")
    print("    • PerformerLayer - universal layer (encoder/decoder)")
    print("    • PerformerEncoder - encoder-only model")
    print("    • PerformerDecoder - decoder with cross-attention")
    print("    • EncoderDecoderTransformer - full seq2seq model")

    print("\n  ⚡ Features:")
    print("    • Self-attention and cross-attention")
    print("    • Linear attention (O(n) complexity)")
    print("    • Causal and bidirectional variants")
    print("    • Multiple normalization options")
    print("    • GLU activation variants")
    print("    • Flexible masking support")

    print("\n  🎯 Use Cases:")
    print("    • Machine Translation (encoder-decoder)")
    print("    • Text Summarization (encoder-decoder)")
    print("    • Language Modeling (decoder-only)")
    print("    • Text Classification (encoder-only)")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
