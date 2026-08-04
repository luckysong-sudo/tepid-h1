# Delta backend qualification

An optimized Delta implementation is not accepted because it compiles or runs. Tepid-H1
requires one machine-readable qualification report covering:

- forward output and final recurrent state;
- input and initial-state gradients;
- every parameter gradient;
- full-sequence versus chunked recurrent execution;
- declared dtype-specific tolerances;
- warmed, alternating-order reference and candidate timing.

CPU compiler-boundary regression:

```bash
tepid-h1 delta-validate \
  --backend eager \
  --device cpu \
  --dtype float32 \
  --sequence-length 4 \
  --iterations 2 \
  --report artifacts/delta-compiler-boundary.json
```

Target CUDA candidate:

```bash
tepid-h1 delta-validate \
  --backend inductor \
  --device cuda \
  --dtype bfloat16 \
  --sequence-length 64 \
  --iterations 20 \
  --target-device-label "declared-target-device" \
  --report artifacts/delta-cuda-inductor.json
```

Shape-level benchmark matrix:

```bash
tepid-h1 delta-benchmark \
  --backend eager \
  --device cpu \
  --dtype float32 \
  --length 4 \
  --length 8 \
  --length 16 \
  --iterations 3 \
  --report artifacts/delta-benchmark-matrix.json
```

The report separates `numerical_passed` from `optimization_qualified`. The latter becomes
true only when all numerical comparisons pass, the candidate uses Inductor on CUDA, a
target-device label is explicitly declared, and measured candidate throughput exceeds the
reference. CPU or eager runs can validate the compiler boundary but cannot claim an
optimized backend.

The benchmark matrix reuses the same numerical qualification path for each sequence
length and records per-shape throughput, speedup, stable case IDs, shape roles, target
evidence flags and qualification-reason aggregates. It is intended as a stable fixture
for comparing future Triton, CUDA or Inductor candidates without turning local CPU timing
into target-hardware evidence.

The current candidate compiles `GatedDeltaMemoryEager`, while
`GatedDeltaMemoryReference` remains the independent oracle. The eager candidate preserves
the same state-dict and recurrent-state layouts, uses batched matrix reads, and combines
erase and write into one algebraically equivalent rank-one update. It is a portable
optimization and an intermediate target for future Triton or custom CUDA work, not a
claim that the Python recurrence is already a production kernel.

## 2026-07-29 CPU qualification signal

Core revision `dc393cb13e2f477ff557625ba16674151928ae57` passed the forward,
final-state, input-gradient, initial-state-gradient, parameter-gradient and chunked
recurrence comparisons in FP32. On the smoke shape with batch size 1, sequence length 64
and 30 alternating warmed iterations, the candidate processed 9,769.5 tokens/second
versus 6,989.4 for the reference, a 1.398x candidate/reference speedup.

This is a local CPU microbenchmark, not target-hardware qualification. The fused candidate
is deployed to the public ZeroGPU Space for the governed CUDA paired rerun; only that
end-to-end result can show whether the operator-level improvement materially changes the
current tiny hybrid-model bottleneck.
