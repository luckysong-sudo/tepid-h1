# Local GPU evidence

This file records the current Windows host GPU preflight result for local Tepid-H1
accelerator work. It is environment evidence only; it is not target-hardware model or
kernel performance evidence.

## 2026-08-01 preflight

Command:

```bash
tepid-h1 gpu-preflight \
  --nvidia-smi "C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"
```

Observed host GPU:

- `nvidia-smi`: found at `C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe`
- GPU: `GeForce MX150`
- Driver: `388.73`
- Memory: `2048 MiB`

Observed Python runtime:

- PyTorch: `2.13.0+cpu`
- CUDA runtime: `null`
- `torch.cuda.is_available()`: `false`
- CUDA device count: `0`

Current blocker:

- The active virtual environment uses a CPU-only PyTorch build, so Tepid-H1 CUDA paths
  cannot execute on the local GPU yet.

Required next actions:

- Install a CUDA-enabled PyTorch build in the active virtual environment.
- Align or upgrade the NVIDIA driver if the selected PyTorch CUDA runtime requires it.
- Rerun `tepid-h1 gpu-preflight` before treating `delta-benchmark --device cuda`,
  `moe-benchmark --device cuda`, or `compare-smoke --device cuda` results as local CUDA
  evidence.
