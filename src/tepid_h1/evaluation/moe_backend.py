from __future__ import annotations

import platform
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from math import isfinite
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
    target_device_label: str | None = None
    router_assignment_cv_threshold: float = 0.25
    minimum_grouped_over_dispatch_speedup: float = 1.0

    def __post_init__(self) -> None:
        batch_size = _validate_moe_int("batch_size", self.batch_size)
        object.__setattr__(self, "batch_size", batch_size)
        iterations = _validate_moe_int("iterations", self.iterations)
        object.__setattr__(self, "iterations", iterations)
        seed = _validate_moe_int("seed", self.seed)
        object.__setattr__(self, "seed", seed)

        if self.variant not in {"smoke", "prototype"}:
            raise ValueError("variant must be 'smoke' or 'prototype'")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16 or float16")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU MoE benchmarking currently requires float32")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(self.sequence_lengths, tuple):
            raise TypeError("sequence_lengths must be a tuple")
        if not self.sequence_lengths:
            raise ValueError("sequence_lengths must not be empty")
        sequence_lengths = tuple(
            _validate_moe_int("sequence_lengths", sequence_length)
            for sequence_length in self.sequence_lengths
        )
        object.__setattr__(self, "sequence_lengths", sequence_lengths)
        for sequence_length in sequence_lengths:
            if not 1 <= sequence_length <= 256:
                raise ValueError("sequence_lengths must be between 1 and 256")
        if not 1 <= iterations <= 100:
            raise ValueError("iterations must be between 1 and 100")
        if self.target_device_label is not None and not self.target_device_label.strip():
            raise ValueError("target_device_label must be non-empty when provided")
        if (
            isinstance(self.router_assignment_cv_threshold, bool)
            or not isfinite(self.router_assignment_cv_threshold)
            or self.router_assignment_cv_threshold < 0
        ):
            raise ValueError("router_assignment_cv_threshold must be finite and non-negative")
        if (
            isinstance(self.minimum_grouped_over_dispatch_speedup, bool)
            or not isfinite(self.minimum_grouped_over_dispatch_speedup)
            or self.minimum_grouped_over_dispatch_speedup < 0
        ):
            raise ValueError(
                "minimum_grouped_over_dispatch_speedup must be finite and non-negative"
            )


def _validate_moe_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def benchmark_routed_moe(config: RoutedMoEBenchmarkConfig) -> dict[str, Any]:
    device, dtype = _resolve_device(config)
    model_config = _model_config(config.variant)
    target_hardware_evidence = device.type == "cuda" and config.target_device_label is not None
    boundary_lengths = _boundary_sequence_lengths(config.sequence_lengths)

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
        router = _router_report(layer, model_config.moe_top_k)
        router_cv_status = _router_cv_status(
            router["assignment_coefficient_of_variation"],
            config.router_assignment_cv_threshold,
        )
        speedup_status = _speedup_status(
            timing["grouped_over_dispatch_speedup"],
            config.minimum_grouped_over_dispatch_speedup,
        )
        cases.append(
            {
                "case_id": f"moe-{config.variant}-{config.device}-{config.dtype}-b"
                f"{config.batch_size}-s{sequence_length}",
                "sequence_length": sequence_length,
                "shape_role": _shape_role(sequence_length, boundary_lengths),
                "batch_tokens": input_tensor.shape[0] * input_tensor.shape[1],
                "batch_size": config.batch_size,
                "device": config.device,
                "dtype": config.dtype,
                "target_device_label": config.target_device_label,
                "target_hardware_evidence": target_hardware_evidence,
                "tokens": timing["tokens"],
                "dispatch_oracle_elapsed_seconds": timing["dispatch_oracle_elapsed_seconds"],
                "grouped_elapsed_seconds": timing["grouped_elapsed_seconds"],
                "elapsed_seconds": timing["grouped_elapsed_seconds"],
                "dispatch_oracle_tokens_per_second": timing["dispatch_oracle_tokens_per_second"],
                "grouped_tokens_per_second": timing["grouped_tokens_per_second"],
                "tokens_per_second": timing["grouped_tokens_per_second"],
                "grouped_over_dispatch_speedup": timing["grouped_over_dispatch_speedup"],
                "minimum_grouped_over_dispatch_speedup": (
                    config.minimum_grouped_over_dispatch_speedup
                ),
                **speedup_status,
                "numerical_passed": numerical_passed,
                "max_abs_error": max_abs_error,
                "router": {
                    **router,
                    "assignment_cv_threshold": config.router_assignment_cv_threshold,
                    "assignment_cv_within_threshold": (
                        router["assignment_coefficient_of_variation"]
                        <= config.router_assignment_cv_threshold
                    ),
                    **router_cv_status,
                },
            }
        )

    throughputs = [case["tokens_per_second"] for case in cases]
    speedups = [case["grouped_over_dispatch_speedup"] for case in cases]
    router_load_cvs = [
        case["router"]["assignment_coefficient_of_variation"] for case in cases
    ]
    max_router_cv_case = _max_router_cv_case(cases)
    min_speedup_case = _min_speedup_case(cases)
    summary_router_cv_status = _router_cv_status(
        max_router_cv_case["router"]["assignment_coefficient_of_variation"],
        config.router_assignment_cv_threshold,
    )
    summary_speedup_status = _speedup_status(
        min_speedup_case["grouped_over_dispatch_speedup"],
        config.minimum_grouped_over_dispatch_speedup,
    )
    shape_roles = _count_values(str(case["shape_role"]) for case in cases)
    all_numerical_passed = all(case["numerical_passed"] for case in cases)
    all_grouped_speedups_meet_threshold = all(
        value >= config.minimum_grouped_over_dispatch_speedup for value in speedups
    )
    all_router_assignment_cv_within_threshold = all(
        value <= config.router_assignment_cv_threshold for value in router_load_cvs
    )
    target_hardware_case_count = sum(1 for case in cases if case["target_hardware_evidence"])
    m4_proxy_status = _m4_moe_proxy_status(
        all_numerical_passed=all_numerical_passed,
        all_grouped_speedups_meet_threshold=all_grouped_speedups_meet_threshold,
        all_router_assignment_cv_within_threshold=all_router_assignment_cv_within_threshold,
        target_hardware_case_count=target_hardware_case_count,
        case_count=len(cases),
    )
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
        "environment": {
            **_environment(device, dtype),
            "cuda_available": device.type == "cuda",
            "target_device_label_declared": config.target_device_label is not None,
            "sequence_length_min": min(config.sequence_lengths),
            "sequence_length_max": max(config.sequence_lengths),
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "all_numerical_passed": all_numerical_passed,
            "target_hardware_case_count": target_hardware_case_count,
            "shape_roles": shape_roles,
            "min_tokens_per_second": min(throughputs),
            "max_tokens_per_second": max(throughputs),
            "min_grouped_over_dispatch_speedup": min(speedups),
            "max_grouped_over_dispatch_speedup": max(speedups),
            "min_grouped_over_dispatch_speedup_case_id": min_speedup_case["case_id"],
            "min_grouped_over_dispatch_speedup_sequence_length": min_speedup_case[
                "sequence_length"
            ],
            "minimum_grouped_over_dispatch_speedup": (
                config.minimum_grouped_over_dispatch_speedup
            ),
            "all_grouped_speedups_meet_threshold": all_grouped_speedups_meet_threshold,
            "grouped_speedup_status": summary_speedup_status["grouped_speedup_status"],
            "grouped_speedup_reason": summary_speedup_status["grouped_speedup_reason"],
            "min_router_assignment_cv": min(router_load_cvs),
            "max_router_assignment_cv": max(router_load_cvs),
            "max_router_assignment_cv_case_id": max_router_cv_case["case_id"],
            "max_router_assignment_cv_sequence_length": max_router_cv_case[
                "sequence_length"
            ],
            "router_assignment_cv_threshold": config.router_assignment_cv_threshold,
            "all_router_assignment_cv_within_threshold": (
                all_router_assignment_cv_within_threshold
            ),
            "router_assignment_cv_status": summary_router_cv_status["assignment_cv_status"],
            "router_assignment_cv_reason": summary_router_cv_status["assignment_cv_reason"],
            **m4_proxy_status,
        },
        "interpretation": (
            "This benchmark compares the grouped selected-expert evaluator against a "
            "per-expert dispatch oracle. It records numerical parity, routing load and "
            "local throughput for future grouped-GEMM or fused-dispatch candidates; it "
            "is not an optimized MoE kernel qualification."
        ),
    }


def _boundary_sequence_lengths(sequence_lengths: tuple[int, ...]) -> tuple[int, int]:
    return min(sequence_lengths), max(sequence_lengths)


def _shape_role(sequence_length: int, boundaries: tuple[int, int]) -> str:
    minimum, maximum = boundaries
    if minimum == maximum:
        return "single"
    if sequence_length == minimum:
        return "minimum"
    if sequence_length == maximum:
        return "maximum"
    return "intermediate"


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _router_cv_status(value: float, threshold: float) -> dict[str, str]:
    if value <= threshold:
        return {
            "assignment_cv_status": "passed",
            "assignment_cv_reason": (
                f"assignment CV {value:.6g} is within threshold {threshold:.6g}"
            ),
        }
    return {
        "assignment_cv_status": "failed",
        "assignment_cv_reason": (
            f"assignment CV {value:.6g} exceeds threshold {threshold:.6g}"
        ),
    }


def _speedup_status(value: float, threshold: float) -> dict[str, str | bool]:
    if value >= threshold:
        return {
            "grouped_speedup_meets_threshold": True,
            "grouped_speedup_status": "passed",
            "grouped_speedup_reason": (
                f"grouped speedup {value:.6g} meets threshold {threshold:.6g}"
            ),
        }
    return {
        "grouped_speedup_meets_threshold": False,
        "grouped_speedup_status": "failed",
        "grouped_speedup_reason": (
            f"grouped speedup {value:.6g} is below threshold {threshold:.6g}"
        ),
    }


def _m4_moe_proxy_status(
    *,
    all_numerical_passed: bool,
    all_grouped_speedups_meet_threshold: bool,
    all_router_assignment_cv_within_threshold: bool,
    target_hardware_case_count: int,
    case_count: int,
) -> dict[str, Any]:
    blockers = []
    if not all_numerical_passed:
        blockers.append("numerical parity failed for one or more cases")
    if target_hardware_case_count != case_count:
        blockers.append(
            f"target-hardware evidence is present for {target_hardware_case_count} "
            f"of {case_count} cases"
        )
    if not all_grouped_speedups_meet_threshold:
        blockers.append("grouped speedup threshold failed for one or more cases")
    if not all_router_assignment_cv_within_threshold:
        blockers.append("router assignment CV threshold failed for one or more cases")
    if blockers:
        return {
            "m4_moe_proxy_passed": False,
            "m4_moe_proxy_status": "blocked",
            "m4_moe_proxy_blockers": blockers,
        }
    return {
        "m4_moe_proxy_passed": True,
        "m4_moe_proxy_status": "passed",
        "m4_moe_proxy_blockers": [],
    }


def _max_router_cv_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        cases,
        key=lambda case: case["router"]["assignment_coefficient_of_variation"],
    )


def _min_speedup_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return min(cases, key=lambda case: case["grouped_over_dispatch_speedup"])


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
        for _ in range(max(1, iterations // 2)):
            function(input_tensor)
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
    variance = sum((count - average_assignments) ** 2 for count in expert_counts) / len(
        expert_counts
    )
    assignment_cv = (variance**0.5) / average_assignments
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
        "assignment_coefficient_of_variation": assignment_cv,
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
