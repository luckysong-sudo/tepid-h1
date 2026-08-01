from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

LEGACY_NVIDIA_DRIVER_MAJOR = 450
SMOKE_ONLY_GPU_MEMORY_MIB = 8192


@dataclass(frozen=True)
class LocalGPUPreflightConfig:
    nvidia_smi_path: str | None = None


def build_local_gpu_preflight_report(
    config: LocalGPUPreflightConfig | None = None,
) -> dict[str, Any]:
    effective_config = config or LocalGPUPreflightConfig()
    nvidia_smi_path = _resolve_nvidia_smi(effective_config.nvidia_smi_path)
    hardware_probe = _hardware_probe(nvidia_smi_path)
    torch_probe = _torch_probe()
    blockers = _blockers(hardware_probe, torch_probe)
    capacity_warnings = _capacity_warnings(hardware_probe)
    readiness = _readiness(blockers, capacity_warnings)
    recommended_actions = _recommended_actions(hardware_probe, torch_probe, blockers)
    validation_plan = _validation_plan(ready_for_cuda=not blockers)
    return {
        "schema_version": 1,
        "experiment": "local_gpu_preflight",
        "config": asdict(effective_config),
        "hardware": hardware_probe,
        "torch": torch_probe,
        "ready_for_cuda": not blockers,
        "blockers": blockers,
        "capacity_warnings": capacity_warnings,
        "readiness": readiness,
        "recommended_actions": recommended_actions,
        "validation_plan": validation_plan,
        "interpretation": (
            "This preflight checks whether the local host can run Tepid-H1 CUDA paths. "
            "A visible NVIDIA GPU is not sufficient; the active Python environment must "
            "also provide a CUDA-enabled PyTorch build that can enumerate the device."
        ),
    }


def _resolve_nvidia_smi(configured_path: str | None) -> Path | None:
    if configured_path:
        path = Path(configured_path)
        return path if path.exists() else None
    discovered = shutil.which("nvidia-smi")
    if discovered is not None:
        return Path(discovered)
    common_path = Path("C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe")
    return common_path if common_path.exists() else None


def _hardware_probe(nvidia_smi_path: Path | None) -> dict[str, Any]:
    if nvidia_smi_path is None:
        return {
            "nvidia_smi_found": False,
            "nvidia_smi_path": None,
            "gpus": [],
            "errors": ["nvidia-smi was not found"],
        }

    command = [
        str(nvidia_smi_path),
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except OSError as error:
        return {
            "nvidia_smi_found": True,
            "nvidia_smi_path": str(nvidia_smi_path),
            "gpus": [],
            "errors": [str(error)],
        }

    errors = []
    if completed.returncode != 0:
        errors.append(completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}")
    return {
        "nvidia_smi_found": True,
        "nvidia_smi_path": str(nvidia_smi_path),
        "gpus": _parse_nvidia_smi_query(completed.stdout),
        "errors": errors,
    }


def _parse_nvidia_smi_query(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, driver_version, memory_total_mib = parts
        try:
            memory_total = int(memory_total_mib)
        except ValueError:
            memory_total = None
        driver_major = _driver_major(driver_version)
        gpus.append(
            {
                "name": name,
                "driver_version": driver_version,
                "driver_major": driver_major,
                "legacy_driver": (
                    driver_major is not None and driver_major < LEGACY_NVIDIA_DRIVER_MAJOR
                ),
                "memory_total_mib": memory_total,
            }
        )
    return gpus


def _driver_major(driver_version: str) -> int | None:
    major, _, _ = driver_version.partition(".")
    try:
        return int(major)
    except ValueError:
        return None


def _torch_probe() -> dict[str, Any]:
    report: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "bf16_supported": False,
        "devices": [],
    }
    if torch.cuda.is_available():
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": properties.total_memory,
                }
            )
        report["devices"] = devices
    return report


def _blockers(
    hardware_probe: dict[str, Any],
    torch_probe: dict[str, Any],
) -> list[str]:
    blockers = []
    if not hardware_probe["gpus"]:
        blockers.append("nvidia-smi did not report a local NVIDIA GPU")
    if not torch_probe["cuda_available"]:
        if torch_probe["cuda_runtime"] is None:
            blockers.append("installed PyTorch build does not include CUDA")
        else:
            blockers.append("PyTorch CUDA runtime cannot enumerate a CUDA device")
    return blockers


def _capacity_warnings(hardware_probe: dict[str, Any]) -> list[str]:
    warnings = []
    for gpu in hardware_probe["gpus"]:
        memory_total_mib = gpu.get("memory_total_mib")
        if isinstance(memory_total_mib, int) and memory_total_mib < SMOKE_ONLY_GPU_MEMORY_MIB:
            warnings.append(
                f"{gpu['name']} reports {memory_total_mib} MiB VRAM; use this device "
                "for smoke and operator-level checks only"
            )
    return warnings


def _readiness(
    blockers: list[str],
    capacity_warnings: list[str],
) -> dict[str, dict[str, Any]]:
    cuda_ready = not blockers
    smoke_status = "ready" if cuda_ready else "blocked"
    smoke_reasons = [] if cuda_ready else blockers
    scale_reasons = [*blockers, *capacity_warnings]
    return {
        "cuda_runtime": {
            "status": "ready" if cuda_ready else "blocked",
            "reasons": [] if cuda_ready else blockers,
        },
        "operator_smoke": {
            "status": smoke_status,
            "reasons": smoke_reasons,
        },
        "training_smoke": {
            "status": smoke_status,
            "reasons": smoke_reasons,
        },
        "scale_training": {
            "status": "blocked" if scale_reasons else "not_assessed",
            "reasons": scale_reasons
            or ["scale training requires a separate target-hardware experiment plan"],
        },
    }


def _recommended_actions(
    hardware_probe: dict[str, Any],
    torch_probe: dict[str, Any],
    blockers: list[str],
) -> list[str]:
    if not blockers:
        return [
            "run tepid-h1 delta-benchmark --device cuda with a target device label",
            "run tepid-h1 moe-benchmark --device cuda to collect reference routing throughput",
        ]

    actions = []
    if not hardware_probe["gpus"]:
        actions.append("install or expose an NVIDIA driver so nvidia-smi reports the GPU")
    if any(gpu.get("legacy_driver") for gpu in hardware_probe["gpus"]):
        actions.append(
            "upgrade or align the NVIDIA driver before installing a modern CUDA-enabled "
            "PyTorch build"
        )
    if torch_probe["cuda_runtime"] is None:
        actions.append("install a CUDA-enabled PyTorch build in the active virtual environment")
    if not torch_probe["cuda_available"] and torch_probe["cuda_runtime"] is not None:
        actions.append("align the NVIDIA driver with the CUDA runtime used by PyTorch")
    if hardware_probe["gpus"] and torch_probe["cuda_runtime"] is None:
        actions.append(
            "rerun gpu-preflight before treating local benchmark results as CUDA evidence"
        )
    return actions


def _validation_plan(*, ready_for_cuda: bool) -> list[dict[str, str]]:
    cuda_status = "ready" if ready_for_cuda else "blocked"
    return [
        {
            "name": "local_gpu_preflight",
            "status": "passed" if ready_for_cuda else "blocked",
            "command": "tepid-h1 gpu-preflight",
            "purpose": "confirm host GPU visibility and PyTorch CUDA readiness",
            "scope": "environment preflight",
        },
        {
            "name": "delta_cuda_benchmark",
            "status": cuda_status,
            "command": (
                "tepid-h1 delta-benchmark --device cuda --dtype float32 "
                "--target-device-label local-gpu --length 4 --length 8 --iterations 3"
            ),
            "purpose": "collect CUDA Delta numerical and shape-level throughput evidence",
            "scope": "operator smoke",
        },
        {
            "name": "moe_cuda_benchmark",
            "status": cuda_status,
            "command": (
                "tepid-h1 moe-benchmark --device cuda --dtype float32 "
                "--length 4 --length 8 --iterations 3"
            ),
            "purpose": "collect CUDA reference MoE routing-load throughput evidence",
            "scope": "operator smoke",
        },
        {
            "name": "paired_cuda_smoke",
            "status": cuda_status,
            "command": (
                "tepid-h1 compare-smoke --steps 1 --trials 1 --device cuda "
                "--dtype float32 --corpus configs/paired_corpus.example.jsonl "
                "--inventory configs/data_inventory.example.json"
            ),
            "purpose": "verify end-to-end governed CUDA measurement plumbing",
            "scope": "training smoke",
        },
    ]
