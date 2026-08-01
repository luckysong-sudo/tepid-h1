"""Deployment adapters that keep external service concerns outside model code."""

from .local_gpu import LocalGPUPreflightConfig, build_local_gpu_preflight_report
from .zero_gpu import ZeroGPUJobConfig, run_zero_gpu_job

__all__ = [
    "LocalGPUPreflightConfig",
    "ZeroGPUJobConfig",
    "build_local_gpu_preflight_report",
    "run_zero_gpu_job",
]
