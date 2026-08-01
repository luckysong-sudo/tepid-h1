# Local GPU evidence

This file records the current Windows host GPU preflight result for local Tepid-H1
accelerator work. It is environment evidence only; it is not target-hardware model or
kernel performance evidence.

## 2026-08-01 preflight

Command:

```bash
tepid-h1 gpu-preflight \
  --nvidia-smi "C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe" \
  --minimum-operator-memory-mib 8192 \
  --minimum-scale-training-memory-mib 24576
```

Observed host GPU:

- `nvidia-smi`: found at `C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe`
- GPU: `GeForce MX150`
- Driver: `388.73`
- Driver major: `388`
- Driver status: legacy for modern CUDA-enabled PyTorch builds
- Memory: `2048 MiB`
- Operator memory threshold: `8192 MiB`
- Scale-training memory threshold: `24576 MiB`
- Capacity scope: below both operator and scale-training thresholds; smoke checks only
  until a larger CUDA device is available

Observed Python runtime:

- PyTorch: `2.13.0+cpu`
- CUDA runtime: `null`
- `torch.cuda.is_available()`: `false`
- CUDA device count: `0`

Current blocker:

- The active virtual environment uses a CPU-only PyTorch build, so Tepid-H1 CUDA paths
  cannot execute on the local GPU yet.
- Readiness summary:
  - CUDA runtime: blocked
  - Operator smoke: blocked until CUDA runtime is enabled
  - Training smoke: blocked until CUDA runtime is enabled
  - Scale training: blocked by both CUDA runtime and MX150 2048 MiB capacity

Required next actions:

- Upgrade or align the NVIDIA driver before installing a modern CUDA-enabled PyTorch
  build.
- Install a CUDA-enabled PyTorch build in the active virtual environment.
- Rerun `tepid-h1 gpu-preflight` before treating `delta-benchmark --device cuda`,
  `moe-benchmark --device cuda`, or `compare-smoke --device cuda` results as local CUDA
  evidence.
- Keep local MX150 runs constrained to small smoke/operator checks; do not use this GPU
  as evidence for M1 350M training, long-window quality, or production kernel readiness.

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
