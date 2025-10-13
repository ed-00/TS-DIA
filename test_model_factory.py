#!/usr/bin/env python
"""
Test script for Model Factory Pattern

Tests:
- Model configuration parsing from YAML
- Model factory creation for all types
- Combined model and dataset parsing
- Parameter validation
"""

import torch

from model.model_factory import ModelFactory, create_model
from model.parse_model_args import parse_model_config
from parse_args import parse_config


def test_encoder_creation():
    """Test encoder-only model creation"""
    print("\n" + "=" * 70)
    print("Testing Encoder-Only Model Creation")
    print("=" * 70)

    config = parse_model_config("configs/encoder_model.yml")

    print("\nConfiguration:")
    print(f"  Type: {config.model_type}")
    print(f"  Name: {config.name}")

    # Create model
    model = create_model(config)

    print("\nModel created:")
    print(f"  Class: {model.__class__.__name__}")
    print(f"  Layers: {len(model.layers)}")

    # Test forward pass
    batch_size = 4
    seq_len = 32
    d_model = 256
    x = torch.randn(batch_size, seq_len, d_model)

    output = model(x)
    print("\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == x.shape
    print("✅ Encoder test passed!")


def test_decoder_creation():
    """Test decoder-only model creation"""
    print("\n" + "=" * 70)
    print("Testing Decoder-Only Model Creation")
    print("=" * 70)

    config = parse_model_config("configs/decoder_model.yml")

    print("\nConfiguration:")
    print(f"  Type: {config.model_type}")
    print(f"  Name: {config.name}")
    print(f"  Cross-attention: {config.config.use_cross_attention}")

    # Create model
    model = create_model(config)

    print("\nModel created:")
    print(f"  Class: {model.__class__.__name__}")
    print(f"  Layers: {len(model.layers)}")

    # Test forward pass with causal masking
    batch_size = 4
    seq_len = 32
    d_model = 512
    x = torch.randn(batch_size, seq_len, d_model)

    output = model(x, use_causal_mask=True)
    print("\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == x.shape
    print("✅ Decoder test passed!")


def test_encoder_decoder_creation():
    """Test encoder-decoder model creation"""
    print("\n" + "=" * 70)
    print("Testing Encoder-Decoder Model Creation")
    print("=" * 70)

    config = parse_model_config("configs/example_model.yml")

    print("\nConfiguration:")
    print(f"  Type: {config.model_type}")
    print(f"  Name: {config.name}")

    # Create model
    model = create_model(config)

    print("\nModel created:")
    print(f"  Class: {model.__class__.__name__}")
    print(f"  Encoder layers: {len(model.encoder.layers)}")
    print(f"  Decoder layers: {len(model.decoder.layers)}")

    # Test forward pass
    batch_size = 4
    src_len = 64
    tgt_len = 48
    d_model = 512

    src = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randn(batch_size, tgt_len, d_model)

    output = model(src, tgt, use_causal_mask=True)

    print("\nForward pass:")
    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == tgt.shape
    print("✅ Encoder-decoder test passed!")


def test_combined_parsing():
    """Test combined model and dataset parsing"""
    print("\n" + "=" * 70)
    print("Testing Combined Model + Dataset Parsing")
    print("=" * 70)

    model_config, dataset_configs = parse_config("configs/full_experiment.yml")

    print("\nModel Configuration:")
    print(f"  Type: {model_config.model_type}")
    print(f"  Name: {model_config.name}")

    if dataset_configs:
        print("\nDataset Configurations:")
        print(f"  Number of datasets: {len(dataset_configs)}")
        for i, config in enumerate(dataset_configs, 1):
            print(f"  {i}. {config.name}")

    # Create model
    model = create_model(model_config)
    print(f"\nModel created: {model.__class__.__name__}")

    print("✅ Combined parsing test passed!")


def test_factory_methods():
    """Test factory class methods"""
    print("\n" + "=" * 70)
    print("Testing Factory Methods")
    print("=" * 70)

    # List available model types
    model_types = ModelFactory.list_model_types()
    print(f"\nAvailable model types: {model_types}")

    # Test validation
    config = parse_model_config("configs/encoder_model.yml")
    config.validate()
    print("✅ Validation passed")

    # Test each factory method
    print("\nTesting direct factory methods:")

    # Encoder
    encoder = ModelFactory.create_encoder(config.config)
    print(f"  ✓ Created encoder: {encoder.__class__.__name__}")

    # Decoder
    decoder_config = parse_model_config("configs/decoder_model.yml")
    decoder = ModelFactory.create_decoder(decoder_config.config)
    print(f"  ✓ Created decoder: {decoder.__class__.__name__}")

    # Encoder-decoder
    enc_dec_config = parse_model_config("configs/example_model.yml")
    enc_dec = ModelFactory.create_encoder_decoder(enc_dec_config.config)
    print(f"  ✓ Created encoder-decoder: {enc_dec.__class__.__name__}")

    print("✅ Factory methods test passed!")


def test_parameter_counts():
    """Test parameter counting for different models"""
    print("\n" + "=" * 70)
    print("Testing Parameter Counts")
    print("=" * 70)

    configs = [
        ("configs/encoder_model.yml", "Encoder"),
        ("configs/decoder_model.yml", "Decoder"),
        ("configs/example_model.yml", "Encoder-Decoder"),
    ]

    for config_path, model_name in configs:
        config = parse_model_config(config_path)
        model = create_model(config)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n{model_name}:")
        print(f"  Total parameters: {num_params:,}")

        # Trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {trainable_params:,}")

    print("\n✅ Parameter count test passed!")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("Model Factory Test Suite")
    print("=" * 70)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    test_encoder_creation()
    test_decoder_creation()
    test_encoder_decoder_creation()
    test_combined_parsing()
    test_factory_methods()
    test_parameter_counts()

    print("\n" + "=" * 70)
    print("✅ All Model Factory tests passed successfully!")
    print("=" * 70)

    print("\n🎉 Your Model Factory Pattern is working perfectly!")
    print("\nYou can now:")
    print("  • Create models from YAML configurations")
    print("  • Use the factory pattern for any model type")
    print("  • Combine model and dataset configs in one file")
    print("  • Extend with new model types easily")

    print("\nExample usage:")
    print("""
    from model.model_factory import create_model
    from model.parse_model_args import parse_model_config

    # Parse configuration
    config = parse_model_config('configs/my_model.yml')

    # Create model
    model = create_model(config)

    # Use model
    output = model(input_tensor)
    """)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
