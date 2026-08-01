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
- Driver major: `388`
- Driver status: legacy for modern CUDA-enabled PyTorch builds
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

- Upgrade or align the NVIDIA driver before installing a modern CUDA-enabled PyTorch
  build.
- Install a CUDA-enabled PyTorch build in the active virtual environment.
- Rerun `tepid-h1 gpu-preflight` before treating `delta-benchmark --device cuda`,
  `moe-benchmark --device cuda`, or `compare-smoke --device cuda` results as local CUDA
  evidence.

Post-enablement validation commands:

```bash
tepid-h1 delta-benchmark \
  --device cuda \
  --dtype float32 \
  --target-device-label local-gpu \
  --length 4 \
  --length 8 \
  --iterations 3

tepid-h1 moe-benchmark \
  --device cuda \
  --dtype float32 \
  --length 4 \
  --length 8 \
  --iterations 3

tepid-h1 compare-smoke \
  --steps 1 \
  --trials 1 \
  --device cuda \
  --dtype float32 \
  --corpus configs/paired_corpus.example.jsonl \
  --inventory configs/data_inventory.example.json
```
