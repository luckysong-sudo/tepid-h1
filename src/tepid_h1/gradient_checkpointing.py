"""Gradient checkpointing support for memory-efficient training."""

from __future__ import annotations

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
    _validate_checkpoint_every(checkpoint_every)
    for index, parent, name, layer in _checkpointable_layers(model):
        if (index + 1) % checkpoint_every == 0:
            parent._modules[name] = CheckpointedLayer(layer, enabled=True)

    return model


class CheckpointedLayer(nn.Module):
    """A layer wrapper that applies gradient checkpointing."""

    def __init__(self, layer: nn.Module, enabled: bool = True) -> None:
        super().__init__()
        self.layer = layer
        self.enabled = enabled

    def forward(self, *args: Any, **kwargs: Any) -> Any:
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
    requested_indices = _validate_layer_indices(layer_indices)
    layers = _checkpointable_layers(model)
    available_indices = {index for index, _, _, _ in layers}
    missing_indices = sorted(requested_indices.difference(available_indices))
    if missing_indices:
        raise ValueError(f"layer_indices are out of range: {missing_indices}")
    for index, parent, name, layer in layers:
        if index in requested_indices:
            parent._modules[name] = CheckpointedLayer(layer, enabled=True)

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
        "total_memory_with_checkpointing_bytes": float(
            param_bytes + activation_memory - saved_memory
        ),
    }


def _validate_checkpoint_every(checkpoint_every: int) -> None:
    if not isinstance(checkpoint_every, int) or isinstance(checkpoint_every, bool):
        raise TypeError("checkpoint_every must be an integer")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")


def _validate_layer_indices(layer_indices: list[int]) -> set[int]:
    indices: set[int] = set()
    for index in layer_indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("layer_indices must contain integers")
        if index < 0:
            raise ValueError("layer_indices must be non-negative")
        indices.add(index)
    return indices


def _checkpointable_layers(
    model: nn.Module,
) -> list[tuple[int, nn.Module, str, nn.Module]]:
    layers: list[tuple[int, nn.Module, str, nn.Module]] = []

    def visit(parent: nn.Module) -> None:
        for name, child in parent.named_children():
            if isinstance(child, CheckpointedLayer):
                continue
            if any(child.children()):
                visit(child)
                continue
            layers.append((len(layers), parent, name, child))

    visit(model)
    return layers
