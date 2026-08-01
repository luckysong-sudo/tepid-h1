"""Gradient checkpointing support for memory-efficient training."""
from __future__ import annotations

from functools import wraps
from typing import Any

import torch
import torch.nn as nn


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_every: int = 2,
) -> nn.Module:
    """Apply gradient checkpointing to a model.

    Args:
        model: The model to apply checkpointing to.
        checkpoint_every: Checkpoint every N layers.

    Returns:
        The model with gradient checkpointing applied.
    """
    checkpoint_counter = [0]

    def make_checkpointable(module: nn.Module) -> nn.Module:
        original_forward = module.forward

        @wraps(original_forward)
        def checkpointed_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            checkpoint_counter[0] += 1
            if checkpoint_counter[0] % checkpoint_every == 0:
                return torch.utils.checkpoint.checkpoint(
                    original_forward,
                    *args,
                    use_reentrant=False,
                    **kwargs,
                )
            return original_forward(*args, **kwargs)

        module.forward = checkpointed_forward
        return module

    # Apply to all modules
    for module in list(model.modules())[1:]:  # Skip the root model
        if hasattr(module, "forward"):
            make_checkpointable(module)

    return model


class CheckpointedLayer(nn.Module):
    """A layer wrapper that applies gradient checkpointing."""

    def __init__(self, layer: nn.Module, enabled: bool = True) -> None:
        super().__init__()
        self.layer = layer
        self.enabled = enabled

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.enabled:
            return torch.utils.checkpoint.checkpoint(
                self.layer,
                *args,
                use_reentrant=False,
                **kwargs,
            )
        return self.layer(*args, **kwargs)

    def extra_repr(self) -> str:
        return f"enabled={self.enabled}"


def wrap_layers_with_checkpointing(
    model: nn.Module,
    layer_indices: list[int],
) -> nn.Module:
    """Wrap specific layers with gradient checkpointing.

    Args:
        model: The model to modify.
        layer_indices: List of layer indices to wrap.

    Returns:
        The model with checkpointing applied to specified layers.
    """
    layer_counter = [0]

    def apply_checkpointing(module: nn.Module) -> nn.Module:
        if hasattr(module, "forward"):
            original_forward = module.forward

            def wrapped_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                current_layer = layer_counter[0]
                layer_counter[0] += 1
                if current_layer in layer_indices:
                    return torch.utils.checkpoint.checkpoint(
                        original_forward,
                        *args,
                        use_reentrant=False,
                        **kwargs,
                    )
                return original_forward(*args, **kwargs)

            module.forward = wrapped_forward
        return module

    for module in list(model.modules())[1:]:  # Skip the root model
        if hasattr(module, "forward"):
            apply_checkpointing(module)

    return model


def estimate_memory_savings(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
) -> dict[str, float]:
    """Estimate memory savings from gradient checkpointing.

    Args:
        model: The model to analyze.
        batch_size: Batch size for estimation.
        sequence_length: Sequence length for estimation.

    Returns:
        Dictionary with memory estimates.
    """
    num_params = sum(p.numel() for p in model.parameters())
    param_bytes = num_params * 2  # Assuming bfloat16

    # Get hidden size from config
    hidden_size = getattr(model.config, "hidden_size", 768)
    num_layers = getattr(model.config, "num_layers", 12)

    # Rough estimate of activation memory
    activation_memory = batch_size * sequence_length * hidden_size * num_layers * 2

    # With checkpointing, we save approximately 50% of activation memory
    saved_memory = activation_memory * 0.5

    return {
        "parameter_memory_bytes": float(param_bytes),
        "estimated_activation_memory_bytes": float(activation_memory),
        "estimated_savings_with_checkpointing_bytes": float(saved_memory),
        "total_memory_with_checkpointing_bytes": float(param_bytes + activation_memory - saved_memory),
    }