"""Model quantization utilities for Tepid-H1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

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
        if isinstance(self.mode, str):
            try:
                object.__setattr__(self, "mode", QuantizationMode(self.mode))
            except ValueError as error:
                raise ValueError(f"unsupported quantization mode: {self.mode}") from error
        elif not isinstance(self.mode, QuantizationMode):
            raise ValueError("mode must be a QuantizationMode")

        if isinstance(self.axis, bool) or self.axis != -1:
            raise ValueError("only axis=-1 is supported for quantization")

        if self.mode == QuantizationMode.NF4:
            if not self.symmetric:
                raise ValueError("NF4 requires symmetric quantization")
            if self.group_size != 0:
                raise ValueError("NF4 uses per-tensor quantization (group_size=0)")
            return
        if self.mode in {QuantizationMode.INT8, QuantizationMode.INT4}:
            if self.group_size <= 0:
                raise ValueError("group_size must be positive for integer quantization")


_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


@dataclass
class QuantizedLayer:
    """A quantized linear layer with metadata."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    scales: torch.Tensor
    zeros: torch.Tensor | None
    original_dtype: torch.dtype
    quantization_mode: QuantizationMode
    original_shape: tuple[int, ...]
    dequantized_cache: torch.Tensor | None = None

    def to(self, device: torch.device) -> "QuantizedLayer":
        return QuantizedLayer(
            weight=self.weight.to(device),
            bias=self.bias.to(device) if self.bias is not None else None,
            scales=self.scales.to(device),
            zeros=self.zeros.to(device) if self.zeros is not None else None,
            original_dtype=self.original_dtype,
            quantization_mode=self.quantization_mode,
            original_shape=self.original_shape,
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
        quantized_weight, scales, zeros = _quantize_tensor(weight, config)
        return cls(
            weight=quantized_weight,
            bias=bias,
            scales=scales,
            zeros=zeros,
            original_dtype=weight.dtype,
            quantization_mode=config.mode,
            original_shape=tuple(weight.shape),
        )

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def dequantize(self) -> torch.Tensor:
        if self.dequantized_cache is not None:
            return self.dequantized_cache
        weight = _dequantize_tensor(
            self.weight,
            self.scales,
            self.zeros,
            self.original_dtype,
            self.quantization_mode,
            self.original_shape,
        )
        self.dequantized_cache = weight
        return weight


def _quantize_tensor(
    weight: torch.Tensor,
    config: QuantizationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        return weight.to(torch.float16), scales, None
    if config.mode == QuantizationMode.BF16:
        scales = torch.ones((), dtype=weight.dtype)
        return weight.to(torch.bfloat16), scales, None
    raise ValueError(f"unsupported quantization mode: {config.mode}")


def _quantize_int8(
    weight: torch.Tensor,
    config: QuantizationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if config.group_size == 0:
        q_min, q_max = (-128, 127) if config.symmetric else (0, 255)
        scales = _per_tensor_scale(weight, q_min, q_max, config.symmetric)
        zeros = None if config.symmetric else torch.zeros((), dtype=weight.dtype)
        quantized = _scale_and_quantize(weight, scales, zeros, q_min, q_max)
        dtype = torch.int8 if config.symmetric else torch.uint8
        return quantized.to(dtype), scales, zeros

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
    quantized = _scale_and_quantize(
        flat,
        scales,
        zeros,
        -128 if config.symmetric else 0,
        127 if config.symmetric else 255,
    )
    dtype = torch.int8 if config.symmetric else torch.uint8
    return (
        quantized.reshape(*shape[:-1], padded_dim)[..., : shape[-1]].to(dtype),
        scales,
        zeros,
    )


def _quantize_int4(
    weight: torch.Tensor,
    config: QuantizationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if config.group_size == 0:
        q_min, q_max = (-8, 7) if config.symmetric else (0, 15)
        scales = _per_tensor_scale(weight, q_min, q_max, config.symmetric)
        zeros = None if config.symmetric else torch.zeros((), dtype=weight.dtype)
        quantized = _scale_and_quantize(weight, scales, zeros, q_min, q_max)
        return _pack_int4(quantized), scales, zeros

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
    quantized = _scale_and_quantize(flat, scales, zeros, -8 if config.symmetric else 0, 15)
    packed = _pack_int4(quantized.reshape(*shape[:-1], padded_dim)[..., : shape[-1]])
    return packed.to(torch.uint8), scales, zeros


def _quantize_nf4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
    flat = weight.float()
    scales = flat.abs().amax().clamp(min=1e-12) / 3.0
    quantized = (flat / scales).round().clamp(-3, 3)
    return _pack_int4(quantized), scales, None


def _quantize_fp8(
    weight: torch.Tensor, config: QuantizationConfig
) -> tuple[torch.Tensor, torch.Tensor, None]:
    flat = weight.float()
    scales = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448.0
    quantized = (flat / scales).clamp(-448.0, 448.0)
    return quantized.to(torch.float8_e4m3fn), scales, None


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
    if weights.shape[-1] % 2 != 0:
        weights = torch.nn.functional.pad(weights, (0, 1))
    low = weights[..., 0::2] & 0x0F
    high = (weights[..., 1::2] & 0x0F) << 4
    return (low | high).to(torch.uint8)


def _unpack_int4(
    packed: torch.Tensor,
    original_shape: tuple[int, ...],
    *,
    signed: bool,
) -> torch.Tensor:
    packed = packed.to(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)
    unpacked = unpacked[..., : original_shape[-1]].to(torch.int16)
    if signed:
        unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    return unpacked.reshape(original_shape).to(torch.float32)


def _dequantize_grouped(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    original_shape: tuple[int, ...],
) -> torch.Tensor:
    if scales.ndim != len(original_shape) + 1 or scales.shape[-1] != 1:
        if zeros is None:
            return quantized.float() * scales
        return (quantized.float() - zeros) * scales

    num_groups = scales.shape[-2]
    group_size = (original_shape[-1] + num_groups - 1) // num_groups
    padded_dim = num_groups * group_size
    values = quantized.float()
    padding = padded_dim - original_shape[-1]
    if padding > 0:
        values = torch.nn.functional.pad(values, (0, padding))
    grouped = values.reshape(*original_shape[:-1], num_groups, group_size)
    if zeros is None:
        dequantized = grouped * scales
    else:
        dequantized = (grouped - zeros) * scales
    return dequantized.reshape(*original_shape[:-1], padded_dim)[..., : original_shape[-1]]


def _dequantize_tensor(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    original_dtype: torch.dtype,
    quantization_mode: QuantizationMode,
    original_shape: tuple[int, ...],
) -> torch.Tensor:
    if quantization_mode in {QuantizationMode.INT4, QuantizationMode.NF4}:
        quantized_values = _unpack_int4(
            quantized,
            original_shape,
            signed=zeros is None,
        )
    else:
        quantized_values = quantized.reshape(original_shape)

    values = _dequantize_grouped(quantized_values, scales, zeros, original_shape)
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
    sizes: dict[str, int] = {
        "weight_bytes": 0,
        "scale_bytes": 0,
        "zero_bytes": 0,
        "bias_bytes": 0,
    }
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pattern in name for pattern in {"skip"}):
            continue
        weight_shape = module.weight.shape
        weight_numel = module.weight.numel()
        last_dim = weight_shape[-1]
        outer_numel = weight_numel // last_dim
        if config.mode in {QuantizationMode.INT8, QuantizationMode.FP8}:
            weight_bytes = weight_numel
        elif config.mode in {QuantizationMode.INT4, QuantizationMode.NF4}:
            weight_bytes = outer_numel * ((last_dim + 1) // 2)
        elif config.mode in {QuantizationMode.FP16, QuantizationMode.BF16}:
            weight_bytes = weight_numel * 2
        else:
            weight_bytes = weight_numel * module.weight.element_size()

        if config.mode in {QuantizationMode.INT8, QuantizationMode.INT4}:
            num_groups = max(
                1,
                (last_dim + config.group_size - 1) // config.group_size,
            )
            scale_count = outer_numel * num_groups
        elif config.mode in {QuantizationMode.NF4, QuantizationMode.FP16, QuantizationMode.BF16}:
            scale_count = 1
        else:
            scale_count = module.weight.shape[0]
        scale_bytes = scale_count * module.weight.element_size()
        zero_bytes = (
            scale_bytes
            if not config.symmetric
            and config.mode in {QuantizationMode.INT8, QuantizationMode.INT4}
            else 0
        )
        sizes["weight_bytes"] += weight_bytes
        sizes["scale_bytes"] += scale_bytes
        sizes["zero_bytes"] += zero_bytes
        if module.bias is not None:
            sizes["bias_bytes"] += module.bias.numel() * module.bias.element_size()
    sizes["total_bytes"] = sum(sizes.values())
    return sizes


def apply_quantized_model(
    model: nn.Module,
    quantized_layers: dict[str, QuantizedLayer],
) -> nn.Module:
    """Replace linear layers with quantized equivalents for inference."""
    resolved_layers: list[tuple[nn.Linear, QuantizedLayer, torch.Tensor]] = []
    for name, quantized in quantized_layers.items():
        current = _resolve_module_path(model, name)
        if not isinstance(current, nn.Linear):
            raise ValueError(f"quantized layer target {name!r} is not an nn.Linear")
        if tuple(current.weight.shape) != quantized.original_shape:
            raise ValueError(
                f"quantized layer {name!r} original_shape {quantized.original_shape!r} "
                f"does not match target weight shape {tuple(current.weight.shape)!r}"
            )
        if (current.bias is None) != (quantized.bias is None):
            raise ValueError(f"quantized layer {name!r} bias presence does not match target")
        if current.bias is not None and quantized.bias is not None:
            if tuple(current.bias.shape) != tuple(quantized.bias.shape):
                raise ValueError(f"quantized layer {name!r} bias shape does not match target")

        dequantized = quantized.dequantize()
        if tuple(dequantized.shape) != tuple(current.weight.shape):
            raise ValueError(f"quantized layer {name!r} dequantized weight shape is invalid")
        resolved_layers.append((current, quantized, dequantized))

    for current, quantized, dequantized in resolved_layers:
        current.weight = nn.Parameter(
            dequantized.to(device=current.weight.device, dtype=current.weight.dtype),
            requires_grad=False,
        )
        if current.bias is not None and quantized.bias is not None:
            current.bias = nn.Parameter(
                quantized.bias.to(device=current.bias.device, dtype=current.bias.dtype),
                requires_grad=False,
            )
    return model


def _resolve_module_path(model: nn.Module, name: str) -> nn.Module:
    current: nn.Module = model
    for part in [p for p in name.split(".") if p]:
        if part.isdigit():
            if not isinstance(current, (nn.Sequential, nn.ModuleList)):
                raise ValueError(f"module path {name!r} indexes non-sequential module")
            index = int(part)
            try:
                current = current[index]
            except IndexError as error:
                raise ValueError(f"module path {name!r} index {index} is out of range") from error
            continue
        if not hasattr(current, part):
            raise ValueError(f"module path {name!r} does not exist")
        child = getattr(current, part)
        if not isinstance(child, nn.Module):
            raise ValueError(f"module path {name!r} resolves through non-module attribute")
        current = child
    return current


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
            "axis": config.axis,
        }

    for name, quantized in quantized_layers.items():
        layer_artifact: dict[str, Any] = {
            "name": name,
            "weight": quantized.weight.detach(),
            "bias": quantized.bias.detach() if quantized.bias is not None else None,
            "scales": quantized.scales.detach(),
            "zeros": quantized.zeros.detach() if quantized.zeros is not None else None,
            "original_shape": quantized.original_shape,
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


def load_quantized_model(
    input_path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, QuantizedLayer]:
    """Load quantized linear layers saved by ``save_quantized_model``."""
    artifact = torch.load(Path(input_path), map_location=map_location, weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("quantized artifact must be a dictionary")
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported quantized artifact schema_version")

    layers = artifact.get("quantized_layers")
    if not isinstance(layers, dict):
        raise ValueError("quantized artifact must contain quantized_layers")

    loaded: dict[str, QuantizedLayer] = {}
    for name, layer_artifact in layers.items():
        if not isinstance(name, str):
            raise ValueError("quantized layer names must be strings")
        if not isinstance(layer_artifact, dict):
            raise ValueError(f"quantized layer {name!r} must be a dictionary")

        mode = _load_quantization_mode(layer_artifact, name)
        original_dtype = _load_original_dtype(layer_artifact, name)
        original_shape = _load_original_shape(layer_artifact, name)
        weight = _load_required_tensor(layer_artifact, "weight", name)
        scales = _load_required_tensor(layer_artifact, "scales", name)
        bias = _load_optional_tensor(layer_artifact, "bias", name)
        zeros = _load_optional_tensor(layer_artifact, "zeros", name)
        dequantized_cache = _load_optional_tensor(layer_artifact, "dequantized", name)

        expected_packed_width = (original_shape[-1] + 1) // 2
        if mode in {QuantizationMode.INT4, QuantizationMode.NF4}:
            expected_weight_shape = (*original_shape[:-1], expected_packed_width)
        else:
            expected_weight_shape = original_shape
        if tuple(weight.shape) != expected_weight_shape:
            raise ValueError(
                f"quantized layer {name!r} weight shape {tuple(weight.shape)!r} "
                f"does not match expected {expected_weight_shape!r}"
            )
        if bias is not None and tuple(bias.shape) != (original_shape[0],):
            raise ValueError(f"quantized layer {name!r} bias shape does not match output dim")
        if dequantized_cache is not None and tuple(dequantized_cache.shape) != original_shape:
            raise ValueError(f"quantized layer {name!r} dequantized cache shape is invalid")

        loaded[name] = QuantizedLayer(
            weight=weight,
            bias=bias,
            scales=scales,
            zeros=zeros,
            original_dtype=original_dtype,
            quantization_mode=mode,
            original_shape=original_shape,
            dequantized_cache=dequantized_cache,
        )
    return loaded


def _load_quantization_mode(layer_artifact: dict[str, Any], name: str) -> QuantizationMode:
    raw_mode = layer_artifact.get("quantization_mode")
    try:
        return QuantizationMode(raw_mode)
    except ValueError as error:
        raise ValueError(f"quantized layer {name!r} has invalid quantization_mode") from error


def _load_original_dtype(layer_artifact: dict[str, Any], name: str) -> torch.dtype:
    raw_dtype = layer_artifact.get("original_dtype")
    if not isinstance(raw_dtype, str) or raw_dtype not in _DTYPE_BY_NAME:
        raise ValueError(f"quantized layer {name!r} has invalid original_dtype")
    return _DTYPE_BY_NAME[raw_dtype]


def _load_original_shape(layer_artifact: dict[str, Any], name: str) -> tuple[int, ...]:
    raw_shape = layer_artifact.get("original_shape")
    if (
        not isinstance(raw_shape, (list, tuple))
        or not raw_shape
        or not all(isinstance(dim, int) and dim > 0 for dim in raw_shape)
    ):
        raise ValueError(f"quantized layer {name!r} has invalid original_shape")
    return tuple(raw_shape)


def _load_required_tensor(
    layer_artifact: dict[str, Any],
    field_name: str,
    layer_name: str,
) -> torch.Tensor:
    value = layer_artifact.get(field_name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"quantized layer {layer_name!r} missing tensor {field_name!r}")
    return value.detach()


def _load_optional_tensor(
    layer_artifact: dict[str, Any],
    field_name: str,
    layer_name: str,
) -> torch.Tensor | None:
    value = layer_artifact.get(field_name)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"quantized layer {layer_name!r} field {field_name!r} must be a tensor")
    return value.detach()
