"""LoRA (Low-Rank Adaptation) adapter for parameter-efficient fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter."""
    r: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"])
    merge_weights: bool = True
    fan_in_fan_out: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.r, int) or isinstance(self.r, bool) or self.r <= 0:
            raise ValueError("r must be a positive integer")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        base_layer: nn.Linear,
        config: LoRAConfig,
        module_name: str = "",
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.module_name = module_name

        self.r = config.r
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout
        self.fan_in_fan_out = config.fan_in_fan_out

        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(config.r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, config.r))
        self.scaling = self.lora_alpha / self.r

        self._copy_weights()

    def _copy_weights(self) -> None:
        """Copy weights from base layer to ensure correctness."""
        if self.fan_in_fan_out:
            self.lora_B.data = self.lora_B.data.T
        with torch.no_grad():
            self.lora_A.zero_()
            self.lora_B.zero_()

    @property
    def weight(self) -> torch.Tensor:
        """Get the full weight (base + LoRA)."""
        if not self.config.merge_weights:
            return self.base_layer.weight
        return self.base_layer.weight + (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        dropout = F.dropout(x, p=self.lora_dropout, training=self.training)
        return base_output + (self.lora_B @ (self.lora_A @ dropout.T)).T * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base_layer.in_features}, "
            f"out_features={self.base_layer.out_features}, "
            f"r={self.r}, alpha={self.lora_alpha}"
        )


class LoRAAdapter(nn.Module):
    """LoRA adapter module that wraps a sub-module with LoRA linear layers."""

    def __init__(
        self,
        module: nn.Module,
        config: LoRAConfig,
        target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.target_modules = target_modules or config.target_modules
        self._apply_lora()

    def _apply_lora(self) -> None:
        """Apply LoRA to target modules."""
        # Collect target modules first to avoid modifying dict during iteration
        targets = []
        for name, child in list(self.module.named_modules()):
            if isinstance(child, nn.Linear) and any(
                target in name for target in self.target_modules
            ):
                targets.append((name, child))
        for name, child in targets:
            lora_linear = LoRALinear(child, self.config, module_name=name)
            setattr(self.module, name, lora_linear)

    def merge_weights(self) -> None:
        """Merge LoRA weights into base weights."""
        for name, child in list(self.module.named_modules()):
            if isinstance(child, LoRALinear) and child.config.merge_weights:
                with torch.no_grad():
                    child.base_layer.weight += (child.lora_B @ child.lora_A) * child.scaling
                # Zero out LoRA matrices after merging (detach first to avoid in-place op on leaf)
                child.lora_A = nn.Parameter(child.lora_A.detach().zero_(), requires_grad=False)
                child.lora_B = nn.Parameter(child.lora_B.detach().zero_(), requires_grad=False)

    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights (restore base + LoRA separation)."""
        for child in self.modules():
            if isinstance(child, LoRALinear):
                child._copy_weights()

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.module(*args, **kwargs)


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: list[str] | None = None,
) -> LoRAAdapter:
    """Apply LoRA adapter to a model.

    Args:
        model: The model to apply LoRA to.
        config: LoRA configuration.
        target_modules: List of module names to target.

    Returns:
        LoRAAdapter wrapping the model.
    """
    return LoRAAdapter(model, config, target_modules)


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all LoRA parameters from a model."""
    params = []
    for child in model.modules():
        if isinstance(child, LoRALinear):
            params.append(child.lora_A)
            params.append(child.lora_B)
    return params


def lora_param_count(model: nn.Module) -> int:
    """Count the number of LoRA parameters in a model."""
    return sum(p.numel() for p in get_lora_params(model))


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all non-LoRA parameters in the model."""
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
