from __future__ import annotations

import platform
import time
from collections.abc import Callable
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
        oracle_output = _dispatch_oracle_output(layer, input_tensor)
        grouped_output = layer(input_tensor)
        max_abs_error = float((oracle_output - grouped_output).detach().float().abs().max())
        numerical_passed = bool(torch.allclose(oracle_output, grouped_output, rtol=1e-5, atol=1e-6))
        timing = _benchmark_pair(
            layer,
            input_tensor,
            iterations=config.iterations,
            device=device,
        )
        cases.append(
            {
                "sequence_length": sequence_length,
                "batch_tokens": input_tensor.shape[0] * input_tensor.shape[1],
                "tokens": timing["tokens"],
                "dispatch_oracle_elapsed_seconds": timing["dispatch_oracle_elapsed_seconds"],
                "grouped_elapsed_seconds": timing["grouped_elapsed_seconds"],
                "elapsed_seconds": timing["grouped_elapsed_seconds"],
                "dispatch_oracle_tokens_per_second": timing["dispatch_oracle_tokens_per_second"],
                "grouped_tokens_per_second": timing["grouped_tokens_per_second"],
                "tokens_per_second": timing["grouped_tokens_per_second"],
                "grouped_over_dispatch_speedup": timing["grouped_over_dispatch_speedup"],
                "numerical_passed": numerical_passed,
                "max_abs_error": max_abs_error,
                "router": _router_report(layer, model_config.moe_top_k),
            }
        )

    throughputs = [case["tokens_per_second"] for case in cases]
    speedups = [case["grouped_over_dispatch_speedup"] for case in cases]
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
            "all_numerical_passed": all(case["numerical_passed"] for case in cases),
            "min_tokens_per_second": min(throughputs),
            "max_tokens_per_second": max(throughputs),
            "min_grouped_over_dispatch_speedup": min(speedups),
            "max_grouped_over_dispatch_speedup": max(speedups),
        },
        "interpretation": (
            "This benchmark compares the grouped selected-expert evaluator against a "
            "per-expert dispatch oracle. It records numerical parity, routing load and "
            "local throughput for future grouped-GEMM or fused-dispatch candidates; it "
            "is not an optimized MoE kernel qualification."
        ),
    }


def _model_config(variant: str) -> TepidH1Config:
    if variant == "smoke":
        return TepidH1Config.smoke()
    if variant == "prototype":
        return TepidH1Config.prototype()
    raise AssertionError(f"unsupported variant: {variant}")


def _benchmark_pair(
    layer: RoutedMoEReference,
    input_tensor: Tensor,
    *,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    layer.eval()
    dispatch_timing = _benchmark_callable(
        lambda tensor: _dispatch_oracle_output(layer, tensor),
        input_tensor,
        iterations=iterations,
        device=device,
    )
    grouped_timing = _benchmark_callable(
        layer,
        input_tensor,
        iterations=iterations,
        device=device,
    )

    return {
        "iterations": iterations,
        "tokens": grouped_timing["tokens"],
        "dispatch_oracle_elapsed_seconds": dispatch_timing["elapsed_seconds"],
        "grouped_elapsed_seconds": grouped_timing["elapsed_seconds"],
        "dispatch_oracle_tokens_per_second": dispatch_timing["tokens_per_second"],
        "grouped_tokens_per_second": grouped_timing["tokens_per_second"],
        "grouped_over_dispatch_speedup": (
            grouped_timing["tokens_per_second"] / dispatch_timing["tokens_per_second"]
        ),
    }


def _benchmark_callable(
    function: Callable[[Tensor], Tensor],
    input_tensor: Tensor,
    *,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    with torch.no_grad():
        function(input_tensor)
    _synchronize(device)

    elapsed = 0.0
    with torch.no_grad():
        for _ in range(iterations):
            _synchronize(device)
            started = time.perf_counter()
            function(input_tensor)
            _synchronize(device)
            elapsed += time.perf_counter() - started

    tokens = input_tensor.shape[0] * input_tensor.shape[1] * iterations
    return {
        "iterations": iterations,
        "tokens": tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
    }


def _dispatch_oracle_output(layer: RoutedMoEReference, input_tensor: Tensor) -> Tensor:
    original_shape = input_tensor.shape
    flat = input_tensor.reshape(-1, original_shape[-1])
    probabilities = layer.router(flat).softmax(dim=-1)
    weights, indices = probabilities.topk(layer.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    routed = torch.zeros_like(flat)
    for expert_index, expert in enumerate(layer.experts):
        token_indices, slots = (indices == expert_index).nonzero(as_tuple=True)
        if token_indices.numel() == 0:
            continue
        expert_output = expert(flat[token_indices])
        expert_output = expert_output * weights[token_indices, slots].unsqueeze(-1)
        routed.index_add_(0, token_indices, expert_output)
    return (layer.shared_expert(flat) + routed).reshape(original_shape)


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
