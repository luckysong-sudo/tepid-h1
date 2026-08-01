"""Mixed precision training utilities for improved performance."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generator

import torch


class PrecisionMode(str, Enum):
    """Training precision modes."""

    FP32 = "fp32"
    BF16 = "bfloat16"
    FP16 = "float16"
    AUTO = "auto"


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""

    enabled: bool = True
    mode: PrecisionMode = PrecisionMode.BF16
    grad_scaler: bool = True
    autocast_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not isinstance(self.grad_scaler, bool):
            raise TypeError("grad_scaler must be a boolean")
        if not isinstance(self.mode, PrecisionMode):
            self.mode = PrecisionMode(self.mode)
        if self.autocast_dtype is None:
            if self.mode == PrecisionMode.BF16:
                self.autocast_dtype = torch.bfloat16
            elif self.mode == PrecisionMode.FP16:
                self.autocast_dtype = torch.float16
            elif self.mode == PrecisionMode.AUTO:
                self.autocast_dtype = _auto_autocast_dtype()
            else:
                self.autocast_dtype = torch.float32
        elif self.autocast_dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise ValueError("autocast_dtype must be float32, bfloat16 or float16")


class MixedPrecisionManager:
    """Manages mixed precision training operations."""

    def __init__(self, config: MixedPrecisionConfig) -> None:
        self.config = config
        self.scaler: torch.amp.GradScaler | None = None
        self._init_scaler()

    def _init_scaler(self) -> None:
        """Initialize gradient scaler if enabled."""
        if (
            self.config.enabled
            and self.config.grad_scaler
            and self.config.autocast_dtype == torch.float16
            and torch.cuda.is_available()
        ):
            self.scaler = torch.amp.GradScaler("cuda")

    @contextmanager
    def autocast_context(self) -> Generator[None, None, None]:
        """Context manager for autocast operations."""
        if not self.config.enabled:
            yield
            return

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = self.config.autocast_dtype
        if dtype == torch.float32:
            yield
            return

        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def unscale_grads(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients after backward pass."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step with scaling."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def to_device(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move tensor to device, handling mixed precision."""
        tensor = tensor.to(device)
        if self.config.enabled and tensor.is_floating_point():
            tensor = tensor.to(dtype=self.config.autocast_dtype)
        return tensor

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for checkpointing."""
        state: dict[str, Any] = {
            "config": {
                "enabled": self.config.enabled,
                "mode": self.config.mode.value,
                "grad_scaler": self.config.grad_scaler,
            }
        }
        if self.scaler is not None:
            state["scaler_state"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state dictionary from checkpoint."""
        config_payload = state.get("config", {})
        if not isinstance(config_payload, dict):
            raise TypeError("mixed precision config state must be a mapping")
        self.config = MixedPrecisionConfig(
            enabled=config_payload.get("enabled", True),
            mode=config_payload.get("mode", PrecisionMode.BF16),
            grad_scaler=config_payload.get("grad_scaler", True),
        )
        self.scaler = None
        self._init_scaler()

        if "scaler_state" in state and self.scaler is not None:
            self.scaler.load_state_dict(state["scaler_state"])


def _auto_autocast_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
