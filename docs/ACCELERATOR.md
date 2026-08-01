# Accelerator experiment contract

The paired experiment runner has an explicit execution-device contract. It never silently
falls back from a requested CUDA device to CPU.

CPU regression:

```bash
tepid-h1 compare-smoke \
  --steps 1 \
  --trials 2 \
  --device cpu \
  --dtype float32 \
  --corpus configs/paired_corpus.example.jsonl \
  --inventory configs/data_inventory.example.json
```

CUDA allocation:

```bash
tepid-h1 compare-smoke \
  --steps 10 \
  --trials 5 \
  --device cuda \
  --dtype bfloat16 \
  --corpus configs/paired_corpus.example.jsonl \
  --inventory configs/data_inventory.example.json \
  --report artifacts/paired-cuda-bf16.json
```

`--device cuda` fails when CUDA is unavailable. BF16 also fails when the allocated device
does not advertise BF16 support. CPU experiments currently require FP32; this avoids
silently comparing materially different operator paths.

Local GPU preflight:

```bash
tepid-h1 gpu-preflight \
  --nvidia-smi "C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe" \
  --report artifacts/local-gpu-preflight.json
```

The preflight report separates host GPU visibility from PyTorch CUDA readiness. On the
current Windows host, `nvidia-smi` reports a GeForce MX150 with driver 388.73, while the
active project environment reports `torch 2.13.0+cpu`, no CUDA runtime and
`torch.cuda.is_available() == false`. That means the local GPU is visible to the OS but
cannot yet execute Tepid-H1 CUDA paths from this virtual environment.

CUDA timing synchronizes the device immediately before and after every measured training
step. Input batches are transferred before timing. Reports identify the GPU, compute
capability, CUDA runtime, total device memory and dtype, and include the maximum PyTorch
allocator memory observed for each model and trial.

On an allocation-based service such as Hugging Face ZeroGPU, the caller must invoke the
runner from inside the service's GPU allocation boundary. This repository does not attempt
to acquire a remote GPU by itself. The same command and report schema can therefore run in
a Space callback, a conventional CUDA runner or a paid training machine without changing
the comparison protocol.

The deployable bundle in `integrations/huggingface-zero-gpu/` follows the official Gradio
and `@spaces.GPU` contract. Copy that directory to the root of a Hugging Face Space, select
ZeroGPU in its hardware settings, and invoke the UI or generated Gradio API endpoint. The
bundle pins the Tepid-H1 core revision, bounds public inputs, binds the synthetic corpus to
its audited digest and returns a downloadable JSON report.

Hugging Face currently documents that ZeroGPU does not support runtime `torch.compile`.
Accordingly, the Space runs the CUDA paired smoke only. `delta-validate --backend inductor`
requires a conventional CUDA runner until an ahead-of-time compiled candidate is provided.

The current smoke configuration is intentionally tiny. A successful CUDA report proves
device compatibility and measurement plumbing only; it is not the M1 350M target-hardware
benchmark.

Reference MoE routing benchmark:

```bash
tepid-h1 moe-benchmark \
  --variant smoke \
  --device cpu \
  --dtype float32 \
  --length 4 \
  --length 8 \
  --length 16 \
  --iterations 3 \
  --report artifacts/moe-benchmark-matrix.json
```

The MoE benchmark records per-shape reference throughput, expert assignment counts,
active expert count and router entropy. It is intended to make future grouped-GEMM or
fused-dispatch candidates comparable against the current correctness-first reference
path; it is not itself an optimized-kernel claim.

The first verified ZeroGPU BF16 execution and its explicit limitations are recorded in
`docs/ZEROGPU_EVIDENCE.md`.
