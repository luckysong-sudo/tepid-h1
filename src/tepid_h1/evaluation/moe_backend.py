from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from ..config import TepidH1Config
from ..modeling.layers import RoutedMoEReference


@dataclass(frozen=True)
class RoutedMoEBenchmarkConfig:
    variant: str = "smoke"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    sequence_lengths: tuple[int, ...] = (4, 8, 16)
    iterations: int = 3
    seed: int = 97

    def __post_init__(self) -> None:
        if self.variant not in {"smoke", "prototype"}:
            raise ValueError("variant must be 'smoke' or 'prototype'")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16 or float16")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU MoE benchmarking currently requires float32")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.sequence_lengths:
            raise ValueError("sequence_lengths must not be empty")
        for sequence_length in self.sequence_lengths:
            if not 1 <= sequence_length <= 256:
                raise ValueError("sequence_lengths must be between 1 and 256")
        if not 1 <= self.iterations <= 100:
            raise ValueError("iterations must be between 1 and 100")


def benchmark_routed_moe(config: RoutedMoEBenchmarkConfig) -> dict[str, Any]:
    device, dtype = _resolve_device(config)
    model_config = _model_config(config.variant)

    torch.manual_seed(config.seed)
    layer = RoutedMoEReference(model_config).to(device=device, dtype=dtype).eval()

    cases: list[dict[str, Any]] = []
    for index, sequence_length in enumerate(config.sequence_lengths):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + index)
        input_cpu = torch.randn(
            config.batch_size,
            sequence_length,
            model_config.hidden_size,
            generator=generator,
        )
        input_tensor = input_cpu.to(device=device, dtype=dtype)
        timing = _benchmark_layer(
            layer,
            input_tensor,
            iterations=config.iterations,
            device=device,
        )
        cases.append(
            {
                "sequence_length": sequence_length,
                "tokens": timing["tokens"],
                "elapsed_seconds": timing["elapsed_seconds"],
                "tokens_per_second": timing["tokens_per_second"],
                "router": _router_report(layer, model_config.moe_top_k),
            }
        )

    throughputs = [case["tokens_per_second"] for case in cases]
    return {
        "schema_version": 1,
        "experiment": "routed_moe_benchmark_matrix",
        "config": asdict(config),
        "model": {
            "variant": config.variant,
            "hidden_size": model_config.hidden_size,
            "num_experts": model_config.moe_num_experts,
            "top_k": model_config.moe_top_k,
            "expert_intermediate_size": model_config.moe_expert_intermediate_size,
            "shared_intermediate_size": model_config.moe_shared_intermediate_size,
        },
        "environment": _environment(device, dtype),
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "min_tokens_per_second": min(throughputs),
            "max_tokens_per_second": max(throughputs),
        },
        "interpretation": (
            "This is a reference MoE dispatch benchmark fixture. It records routing load "
            "and local throughput for future grouped-GEMM or fused-dispatch candidates; "
            "it is not an optimized MoE kernel qualification."
        ),
    }


def _model_config(variant: str) -> TepidH1Config:
    if variant == "smoke":
        return TepidH1Config.smoke()
    if variant == "prototype":
        return TepidH1Config.prototype()
    raise AssertionError(f"unsupported variant: {variant}")


def _benchmark_layer(
    layer: RoutedMoEReference,
    input_tensor: Tensor,
    *,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    layer.eval()
    with torch.no_grad():
        layer(input_tensor)
    _synchronize(device)

    elapsed = 0.0
    with torch.no_grad():
        for _ in range(iterations):
            _synchronize(device)
            started = time.perf_counter()
            layer(input_tensor)
            _synchronize(device)
            elapsed += time.perf_counter() - started

    tokens = input_tensor.shape[0] * input_tensor.shape[1] * iterations
    return {
        "iterations": iterations,
        "tokens": tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
    }


def _router_report(layer: RoutedMoEReference, top_k: int) -> dict[str, Any]:
    stats = layer.last_router_stats
    if stats is None:
        raise RuntimeError("MoE router stats were not produced")
    expert_counts = [int(value) for value in stats.expert_counts.detach().cpu().tolist()]
    assignment_count = sum(expert_counts)
    expected_assignments = stats.router_probabilities.shape[0] * top_k
    average_assignments = assignment_count / len(expert_counts)
    probabilities = stats.router_probabilities.detach().float()
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return {
        "expert_counts": expert_counts,
        "total_assignments": assignment_count,
        "expected_assignments": expected_assignments,
        "active_experts": sum(1 for count in expert_counts if count > 0),
        "max_expert_assignments": max(expert_counts),
        "min_expert_assignments": min(expert_counts),
        "max_over_average_assignments": max(expert_counts) / average_assignments,
        "mean_router_entropy": float(entropy.mean()),
    }


def _resolve_device(config: RoutedMoEBenchmarkConfig) -> tuple[torch.device, torch.dtype]:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if (
        config.device == "cuda"
        and config.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 was requested but the CUDA device does not support it")
    return torch.device(config.device), {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]


def _environment(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device_type": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update(
            {
                "device_name": properties.name,
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": properties.total_memory,
                "cuda_runtime": torch.version.cuda,
            }
        )
    return report


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
