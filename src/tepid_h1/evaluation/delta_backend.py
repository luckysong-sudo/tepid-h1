from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..config import TepidH1Config
from ..modeling.layers import GatedDeltaMemoryEager, GatedDeltaMemoryReference


@dataclass(frozen=True)
class DeltaBackendValidationConfig:
    backend: str = "eager"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    sequence_length: int = 4
    iterations: int = 3
    seed: int = 71
    target_device_label: str | None = None
    verify_gradients: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"eager", "inductor"}:
            raise ValueError("backend must be 'eager' or 'inductor'")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16 or float16")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU Delta validation currently requires float32")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 2 <= self.sequence_length <= 8192:
            raise ValueError("sequence_length must be between 2 and 8192")
        if not 1 <= self.iterations <= 100:
            raise ValueError("iterations must be between 1 and 100")
        if self.target_device_label is not None and not self.target_device_label.strip():
            raise ValueError("target_device_label must be non-empty when provided")


def validate_delta_backend(config: DeltaBackendValidationConfig) -> dict[str, Any]:
    device, dtype = _resolve_device(config)
    tolerance = _tolerance(dtype)
    model_config = TepidH1Config.smoke()

    torch.manual_seed(config.seed)
    reference = GatedDeltaMemoryReference(model_config).to(device=device, dtype=dtype)
    candidate_layer = GatedDeltaMemoryEager(model_config).to(device=device, dtype=dtype)
    candidate_layer.load_state_dict(reference.state_dict())
    candidate = torch.compile(candidate_layer, backend=config.backend, fullgraph=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 1)
    input_cpu = torch.randn(
        config.batch_size,
        config.sequence_length,
        model_config.hidden_size,
        generator=generator,
    )
    state_cpu = torch.randn(
        config.batch_size,
        model_config.num_query_heads,
        model_config.head_dim,
        model_config.head_dim,
        generator=generator,
    )
    reference_input = input_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    candidate_input = input_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    reference_state = state_cpu.to(device=device, dtype=dtype).requires_grad_(True)
    candidate_state = state_cpu.to(device=device, dtype=dtype).requires_grad_(True)

    reference_output, reference_final_state = reference(reference_input, reference_state)
    candidate_output, candidate_final_state = candidate(candidate_input, candidate_state)
    comparisons: dict[str, Any] = {
        "forward_output": _compare(
            reference_output,
            candidate_output,
            **tolerance,
        ),
        "final_state": _compare(
            reference_final_state,
            candidate_final_state,
            **tolerance,
        ),
    }
    if config.verify_gradients:
        _objective(reference_output, reference_final_state).backward()
        _objective(candidate_output, candidate_final_state).backward()
        comparisons.update(
            {
                "input_gradient": _compare(
                    _required_gradient(reference_input, "reference input"),
                    _required_gradient(candidate_input, "candidate input"),
                    **tolerance,
                ),
                "initial_state_gradient": _compare(
                    _required_gradient(reference_state, "reference state"),
                    _required_gradient(candidate_state, "candidate state"),
                    **tolerance,
                ),
                "parameter_gradients": _compare_parameter_gradients(
                    reference,
                    candidate_layer,
                    **tolerance,
                ),
            }
        )
    comparisons["chunked_recurrence"] = _compare_chunked(
        reference,
        candidate,
        input_cpu.to(device=device, dtype=dtype),
        state_cpu.to(device=device, dtype=dtype),
        **tolerance,
    )
    numerical_passed = all(_comparison_passed(item) for item in comparisons.values())

    timing = _benchmark_pair(
        reference,
        candidate,
        input_cpu.to(device=device, dtype=dtype),
        state_cpu.to(device=device, dtype=dtype),
        iterations=config.iterations,
        device=device,
    )
    target_hardware_evidence = (
        device.type == "cuda"
        and config.backend == "inductor"
        and config.target_device_label is not None
    )
    optimization_qualified = (
        numerical_passed
        and target_hardware_evidence
        and timing["candidate_over_reference_speedup"] > 1.0
    )
    return {
        "schema_version": 1,
        "experiment": "delta_backend_qualification",
        "config": asdict(config),
        "implementations": {
            "reference": "GatedDeltaMemoryReference",
            "candidate": "GatedDeltaMemoryEager",
        },
        "environment": _environment(device, dtype),
        "tolerance": tolerance,
        "comparisons": comparisons,
        "numerical_passed": numerical_passed,
        "timing": timing,
        "qualification": {
            "target_hardware_evidence": target_hardware_evidence,
            "optimization_qualified": optimization_qualified,
            "reason": _qualification_reason(
                numerical_passed,
                target_hardware_evidence,
                timing["candidate_over_reference_speedup"],
            ),
        },
        "interpretation": (
            "The eager backend validates the compiler boundary only. An optimized backend "
            "is qualified only after numerical parity and positive speedup on a declared "
            "CUDA target device."
        ),
    }


def _objective(output: Tensor, state: Tensor) -> Tensor:
    return output.float().square().mean() + state.float().square().mean()


def _required_gradient(tensor: Tensor, name: str) -> Tensor:
    if tensor.grad is None:
        raise RuntimeError(f"{name} gradient was not produced")
    return tensor.grad


def _compare_parameter_gradients(
    reference: nn.Module,
    candidate: nn.Module,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if reference_parameters.keys() != candidate_parameters.keys():
        raise ValueError("candidate parameter names do not match the reference")
    details: dict[str, Any] = {}
    for name, reference_parameter in reference_parameters.items():
        candidate_parameter = candidate_parameters[name]
        details[name] = _compare(
            _required_gradient(reference_parameter, f"reference parameter {name}"),
            _required_gradient(candidate_parameter, f"candidate parameter {name}"),
            rtol=rtol,
            atol=atol,
        )
    return {
        "passed": all(item["passed"] for item in details.values()),
        "parameters": details,
    }


def _compare_chunked(
    reference: nn.Module,
    candidate: nn.Module,
    input_tensor: Tensor,
    initial_state: Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    split = input_tensor.shape[1] // 2
    with torch.no_grad():
        reference_output, reference_state = reference(input_tensor, initial_state)
        first_output, state = candidate(input_tensor[:, :split], initial_state)
        second_output, chunked_state = candidate(input_tensor[:, split:], state)
    chunked_output = torch.cat((first_output, second_output), dim=1)
    output_comparison = _compare(reference_output, chunked_output, rtol=rtol, atol=atol)
    state_comparison = _compare(reference_state, chunked_state, rtol=rtol, atol=atol)
    return {
        "passed": output_comparison["passed"] and state_comparison["passed"],
        "output": output_comparison,
        "final_state": state_comparison,
        "split_index": split,
    }


def _compare(
    reference: Tensor,
    candidate: Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    absolute = (reference_float - candidate_float).abs()
    relative = absolute / reference_float.abs().clamp_min(atol)
    return {
        "passed": bool(torch.allclose(reference_float, candidate_float, rtol=rtol, atol=atol)),
        "max_absolute_error": float(absolute.max()),
        "max_relative_error": float(relative.max()),
    }


def _comparison_passed(value: dict[str, Any]) -> bool:
    return bool(value["passed"])


def _benchmark_pair(
    reference: nn.Module,
    candidate: nn.Module,
    input_tensor: Tensor,
    initial_state: Tensor,
    *,
    iterations: int,
    device: torch.device,
) -> dict[str, float | int]:
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        reference(input_tensor, initial_state)
        candidate(input_tensor, initial_state)
    _synchronize(device)

    elapsed = {"reference": 0.0, "candidate": 0.0}
    modules = {"reference": reference, "candidate": candidate}
    with torch.no_grad():
        for iteration in range(iterations):
            order = ("reference", "candidate") if iteration % 2 == 0 else ("candidate", "reference")
            for name in order:
                _synchronize(device)
                started = time.perf_counter()
                modules[name](input_tensor, initial_state)
                _synchronize(device)
                elapsed[name] += time.perf_counter() - started

    tokens = input_tensor.shape[0] * input_tensor.shape[1] * iterations
    reference_throughput = tokens / elapsed["reference"]
    candidate_throughput = tokens / elapsed["candidate"]
    return {
        "iterations": iterations,
        "tokens": tokens,
        "reference_elapsed_seconds": elapsed["reference"],
        "candidate_elapsed_seconds": elapsed["candidate"],
        "reference_tokens_per_second": reference_throughput,
        "candidate_tokens_per_second": candidate_throughput,
        "candidate_over_reference_speedup": candidate_throughput / reference_throughput,
    }


def _resolve_device(
    config: DeltaBackendValidationConfig,
) -> tuple[torch.device, torch.dtype]:
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


def _tolerance(dtype: torch.dtype) -> dict[str, float]:
    return {
        torch.float32: {"rtol": 1e-4, "atol": 1e-5},
        torch.bfloat16: {"rtol": 5e-2, "atol": 5e-2},
        torch.float16: {"rtol": 1e-2, "atol": 1e-2},
    }[dtype]


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


def _qualification_reason(
    numerical_passed: bool,
    target_hardware_evidence: bool,
    speedup: float,
) -> str:
    if not numerical_passed:
        return "numerical parity failed"
    if not target_hardware_evidence:
        return "target CUDA device with the inductor backend was not declared"
    if speedup <= 1.0:
        return "candidate did not produce positive target-hardware speedup"
    return "numerical parity and positive declared target-hardware speedup passed"
