"""Deployment adapters that keep external service concerns outside model code."""

from .zero_gpu import ZeroGPUJobConfig, run_zero_gpu_job

__all__ = ["ZeroGPUJobConfig", "run_zero_gpu_job"]
