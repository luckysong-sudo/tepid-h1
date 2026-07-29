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

CUDA timing synchronizes the device immediately before and after every measured training
step. Input batches are transferred before timing. Reports identify the GPU, compute
capability, CUDA runtime, total device memory and dtype, and include the maximum PyTorch
allocator memory observed for each model and trial.

On an allocation-based service such as Hugging Face ZeroGPU, the caller must invoke the
runner from inside the service's GPU allocation boundary. This repository does not attempt
to acquire a remote GPU by itself. The same command and report schema can therefore run in
a Space callback, a conventional CUDA runner or a paid training machine without changing
the comparison protocol.

The current smoke configuration is intentionally tiny. A successful CUDA report proves
device compatibility and measurement plumbing only; it is not the M1 350M target-hardware
benchmark.
