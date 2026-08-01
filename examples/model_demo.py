"""Demonstration of Tepid-H1 model capabilities."""
from __future__ import annotations

import torch

from tepid_h1 import (
    GQAAttentionNative,
    GQAAttentionReference,
    GatedDeltaMemoryEager,
    GatedDeltaMemoryReference,
    TepidH1CausalLM,
    TepidH1Config,
)


def demo_model_shapes() -> None:
    """Demo model forward pass with shape verification."""
    config = TepidH1Config.prototype()
    model = TepidH1CausalLM(config)

    # Create dummy input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    output = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {output.logits.shape}")
    print(f"Loss: {output.loss.item():.4f}")


def demo_attention_layers() -> None:
    """Demo different attention layer types."""
    config = TepidH1Config.prototype()

    # Delta memory layer
    delta_ref = GatedDeltaMemoryReference(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
    )
    delta_eager = GatedDeltaMemoryEager(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
    )

    # Attention layers
    attn_ref = GQAAttentionReference(
        config=config,
        layer_index=0,
    )
    attn_native = GQAAttentionNative(
        config=config,
        layer_index=0,
    )

    print("Delta layer shapes:")
    print(f"  Reference: {delta_ref}")
    print(f"  Eager: {delta_eager}")
    print("\nAttention layer shapes:")
    print(f"  Reference: {attn_ref}")
    print(f"  Native: {attn_native}")


def demo_gradient_checkpointing() -> None:
    """Demo gradient checkpointing."""
    from tepid_h1 import apply_gradient_checkpointing, TepidH1Config

    config = TepidH1Config.prototype()
    model = TepidH1CausalLM(config)

    # Apply gradient checkpointing
    apply_gradient_checkpointing(model)
    print("Gradient checkpointing applied successfully")

    # Estimate memory savings
    from tepid_h1 import estimate_memory_savings
    savings = estimate_memory_savings(model, config)
    print(f"Estimated memory savings: {savings:.2f}%")


def demo_mixed_precision() -> None:
    """Demo mixed precision training."""
    from tepid_h1 import MixedPrecisionConfig, MixedPrecisionManager, TepidH1Config

    config = TepidH1Config.prototype()
    manager = MixedPrecisionManager(MixedPrecisionConfig.autocast())

    print("Mixed precision manager created:")
    print(f"  Mode: {manager.mode.value}")
    print(f"  Enabled: {manager.is_enabled}")


if __name__ == "__main__":
    print("=" * 50)
    print("Tepid-H1 Model Demo")
    print("=" * 50)

    print("\n1. Model Forward Pass")
    demo_model_shapes()

    print("\n2. Attention Layer Types")
    demo_attention_layers()

    print("\n3. Gradient Checkpointing")
    demo_gradient_checkpointing()

    print("\n4. Mixed Precision")
    demo_mixed_precision()

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)