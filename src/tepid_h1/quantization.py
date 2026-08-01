"""Model quantization utilities for Tepid-H1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn


class QuantizationMode(str, Enum):
    """Quantization precision modes."""

    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bfloat16"


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for model quantization."""

    mode: QuantizationMode = QuantizationMode.INT8
    group_size: int = 128
    symmetric: bool = True
    axis: int = -1

    def __post_init__(self) -> None:
        if self.mode == QuantizationMode.NF4:
            if not self.symmetric:
                raise ValueError("NF4 requires symmetric quantization")
            if self.group_size != 0:
                raise ValueError("NF4 uses per-tensor quantization (group_size=0)")
            return
        if self.mode in {QuantizationMode.INT8, QuantizationMode.INT4}:
            if self.group_size <= 0:
                raise ValueError("group_size must be positive for integer quantization")


@dataclass
class QuantizedLayer:
    """A quantized linear layer with metadata."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    scales: torch.Tensor
    zeros: torch.Tensor | None
    original_dtype: torch.dtype
    quantization_mode: QuantizationMode
    dequantized_cache: torch.Tensor | None = None

    def to(self, device: torch.device) -> "QuantizedLayer":
        return QuantizedLayer(
            weight=self.weight.to(device),
            bias=self.bias.to(device) if self.bias is not None else None,
            scales=self.scales.to(device),
            zeros=self.zeros.to(device) if self.zeros is not None else None,
            original_dtype=self.original_dtype,
            quantization_mode=self.quantization_mode,
            dequantized_cache=None,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: QuantizationConfig,
    ) -> "QuantizedLayer":
        weight = linear.weight.detach()
        bias = linear.bias.detach() if linear.bias is not None else None
        scales, zeros = _quantize_tensor(weight, config)
        return cls(
            weight=weight,
            bias=bias,
            scales=scales,
            zeros=zeros,
            original_dtype=weight.dtype,
            quantization_mode=config.mode,
        )

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def dequantize(self) -> torch.Tensor:
        if self.dequantized_cache is not None:
            return self.dequantized_cache
        weight = _dequantize_tensor(self.weight, self.scales, self.zeros, self.original_dtype)
        self.dequantized_cache = weight
        return weight


def _quantize_tensor(
    weight: torch.Tensor,
    config: QuantizationConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if config.mode == QuantizationMode.INT8:
        return _quantize_int8(weight, config)
    if config.mode == QuantizationMode.INT4:
        return _quantize_int4(weight, config)
    if config.mode == QuantizationMode.NF4:
        return _quantize_nf4(weight)
    if config.mode == QuantizationMode.FP8:
        return _quantize_fp8(weight, config)
    if config.mode == QuantizationMode.FP16:
        scales = torch.ones((), dtype=weight.dtype)
        return (weight.to(torch.float16), scales)
    if config.mode == QuantizationMode.BF16:
        scales = torch.ones((), dtype=weight.dtype)
        return (weight.to(torch.bfloat16), scales)
    raise ValueError(f"unsupported quantization mode: {config.mode}")


def _quantize_int8(
    weight: torch.Tensor,
    config: QuantizationConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if config.group_size == 0:
        q_min, q_max = (-128, 127) if config.symmetric else (0, 255)
        scales = _per_tensor_scale(weight, q_min, q_max, config.symmetric)
        zeros = None if config.symmetric else torch.zeros((), dtype=weight.dtype)
        quantized = _scale_and_quantize(weight, scales, zeros, q_min, q_max)
        return quantized.to(torch.int8), zeros

    flat = weight.float()
    shape = flat.shape
    group_size = max(1, config.group_size)
    num_groups = max(1, (shape[-1] + group_size - 1) // group_size)
    padded_dim = num_groups * group_size
    padding = padded_dim - shape[-1]
    if padding > 0:
        flat = torch.nn.functional.pad(flat, (0, padding))
    flat = flat.reshape(*shape[:-1], num_groups, group_size)
    if config.symmetric:
        scales = flat.abs().amax(dim=-1).clamp(min=1e-12) / 127.0
        zeros = None
    else:
        q_max_float = 255.0
        fmin = flat.amin(dim=-1)
        fmax = flat.amax(dim=-1)
        scales = (fmax - fmin).clamp(min=1e-12) / q_max_float
        zeros = (q_max_float * fmin / (fmin - fmax)).clamp(min=0.0, max=q_max_float)

    scales = scales.reshape(*shape[:-1], num_groups, 1)
    zeros = zeros.reshape(*shape[:-1], num_groups, 1) if zeros is not None else None
    quantized = _scale_and_quantize(flat, scales, zeros, -128 if config.symmetric else 0, 127)
    return quantized.reshape(*shape[:-1], padded_dim)[..., : shape[-1]].to(torch.int8), zeros


def _quantize_int4(
    weight: torch.Tensor,
    config: QuantizationConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if config.group_size == 0:
        q_min, q_max = (-8, 7) if config.symmetric else (0, 15)
        scales = _per_tensor_scale(weight, q_min, q_max, config.symmetric)
        zeros = None if config.symmetric else torch.zeros((), dtype=weight.dtype)
        quantized = _scale_and_quantize(weight, scales, zeros, q_min, q_max)
        return quantized.to(torch.int8), zeros

    flat = weight.float()
    shape = flat.shape
    group_size = max(1, config.group_size)
    num_groups = max(1, (shape[-1] + group_size - 1) // group_size)
    padded_dim = num_groups * group_size
    padding = padded_dim - shape[-1]
    if padding > 0:
        flat = torch.nn.functional.pad(flat, (0, padding))
    flat = flat.reshape(*shape[:-1], num_groups, group_size)
    if config.symmetric:
        scales = flat.abs().amax(dim=-1).clamp(min=1e-12) / 7.0
        zeros = None
    else:
        q_max_float = 15.0
        fmin = flat.amin(dim=-1)
        fmax = flat.amax(dim=-1)
        scales = (fmax - fmin).clamp(min=1e-12) / q_max_float
        zeros = (q_max_float * fmin / (fmin - fmax)).clamp(min=0.0, max=q_max_float)

    scales = scales.reshape(*shape[:-1], num_groups, 1)
    zeros = zeros.reshape(*shape[:-1], num_groups, 1) if zeros is not None else None
    quantized = _scale_and_quantize(flat, scales, zeros, -8 if config.symmetric else 0, 7)
    packed = _pack_int4(quantized.reshape(*shape[:-1], padded_dim)[..., : shape[-1]])
    return packed.to(torch.uint8), zeros


def _quantize_nf4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = weight.float()
    scales = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 3.0
    quantized = (flat / scales).round().clamp(-3, 3)
    return quantized.to(torch.int8), scales


def _quantize_fp8(
    weight: torch.Tensor, config: QuantizationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = weight.float()
    scales = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
    quantized = (flat / scales).clamp(-448.0, 448.0)
    return quantized.to(torch.float8_e4m3fn), scales


def _per_tensor_scale(
    weight: torch.Tensor,
    q_min: float,
    q_max: float,
    symmetric: bool,
) -> torch.Tensor:
    if symmetric:
        scale = weight.abs().amax().clamp(min=1e-12) / abs(q_max)
    else:
        fmin = weight.amin()
        fmax = weight.amax()
        scale = (fmax - fmin).clamp(min=1e-12) / (q_max - q_min)
    return scale


def _scale_and_quantize(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    q_min: int,
    q_max: int,
) -> torch.Tensor:
    if zeros is None:
        quantized = (weight / scales).round().clamp(q_min, q_max)
    else:
        quantized = ((weight / scales) + zeros).round().clamp(q_min, q_max)
    return quantized


def _pack_int4(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(torch.int8)
    even = weights[::2] & 0x0F
    odd = (weights[1::2] & 0x0F) << 4
    return even | odd


def _dequantize_tensor(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    original_dtype: torch.dtype,
) -> torch.Tensor:
    if quantized.dtype == torch.uint8 and quantized.ndim > 1 and quantized.shape[-1] == 0:
        return torch.zeros(*quantized.shape[:-1], 0, dtype=original_dtype)

    if zeros is None:
        values = quantized.float() * scales
    else:
        values = (quantized.float() - zeros) * scales
    return values.to(original_dtype)


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
    *,
    skip_layers: set[str] | None = None,
) -> dict[str, QuantizedLayer]:
    """Quantize a model and return mapping of layer names to quantized layers."""
    skip = skip_layers or set()
    quantized: dict[str, QuantizedLayer] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pattern in name for pattern in skip):
            continue
        quantized[name] = QuantizedLayer.from_linear(module, config)
    return quantized


def estimate_quantized_size(
    model: nn.Module,
    config: QuantizationConfig,
) -> dict[str, int]:
    """Estimate the quantized model size in bytes."""
    sizes: dict[str, int] = {"weight_bytes": 0, "scale_bytes": 0, "zero_bytes": 0}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pattern in name for pattern in {"skip"}):
            continue
        weight_bytes = module.weight.numel()
        if config.mode in {QuantizationMode.INT8, QuantizationMode.FP8}:
            weight_bytes *= 1
        elif config.mode in {QuantizationMode.INT4, QuantizationMode.NF4}:
            weight_bytes = (weight_bytes + 1) // 2
        elif config.mode in {QuantizationMode.FP16, QuantizationMode.BF16}:
            weight_bytes *= 2
        bias_bytes = module.bias.numel() if module.bias is not None else 0
        scale_bytes = max(1, module.weight.shape[-1] // max(config.group_size, 1))
        zero_bytes = (
            scale_bytes
            if not config.symmetric
            and config.mode in {QuantizationMode.INT8, QuantizationMode.INT4}
            else 0
        )
        sizes["weight_bytes"] += weight_bytes
        sizes["scale_bytes"] += scale_bytes
        sizes["zero_bytes"] += zero_bytes
        if bias_bytes > 0:
            sizes["bias_bytes"] = sizes.get("bias_bytes", 0) + bias_bytes
    return sizes


def apply_quantized_model(
    model: nn.Module,
    quantized_layers: dict[str, QuantizedLayer],
) -> nn.Module:
    """Replace linear layers with quantized equivalents for inference."""
    for name, quantized in quantized_layers.items():
        parts = [p for p in name.split(".") if p]
        current: nn.Module = model
        for part in parts:
            if part.isdigit():
                current = cast(nn.Sequential, current)[int(part)]
            else:
                current = getattr(current, part)
        if isinstance(current, nn.Linear):
            dequantized = quantized.dequantize()
            current.weight = nn.Parameter(dequantized, requires_grad=False)
            if quantized.bias is not None:
                current.bias = nn.Parameter(quantized.bias, requires_grad=False)
            else:
                current.register_parameter("bias", None)
    return model


def save_quantized_model(
    model: nn.Module,
    quantized_layers: dict[str, QuantizedLayer],
    output_path: str | Path,
    *,
    config: QuantizationConfig | None = None,
) -> dict[str, Any]:
    """Save a quantized model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Any] = {
        "schema_version": 1,
        "quantization_mode": config.mode.value if config else "none",
        "quantized_layers": {},
        "original_dtype": str(next(model.parameters()).dtype),
    }
    if config is not None:
        artifacts["config"] = {
            "mode": config.mode.value,
            "group_size": config.group_size,
            "symmetric": config.symmetric,
        }

    for name, quantized in quantized_layers.items():
        layer_artifact: dict[str, Any] = {
            "name": name,
            "original_dtype": str(quantized.original_dtype),
            "quantization_mode": quantized.quantization_mode.value,
        }
        if quantized.dequantized_cache is None:
            layer_artifact["dequantized"] = None
        else:
            layer_artifact["dequantized"] = quantized.dequantized_cache.detach()
        artifacts["quantized_layers"][name] = layer_artifact

    torch.save(artifacts, output_path)
    return artifacts
