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

The report separates `numerical_passed` from `optimization_qualified`. The latter becomes
true only when all numerical comparisons pass, the candidate uses Inductor on CUDA, a
target-device label is explicitly declared, and measured candidate throughput exceeds the
reference. CPU or eager runs can validate the compiler boundary but cannot claim an
optimized backend.

The current candidate compiles `GatedDeltaMemoryEager`, while
`GatedDeltaMemoryReference` remains the independent oracle. The eager candidate preserves
the same state-dict and recurrent-state layouts, uses batched matrix reads, and combines
erase and write into one algebraically equivalent rank-one update. It is a portable
optimization and an intermediate target for future Triton or custom CUDA work, not a
claim that the Python recurrence is already a production kernel.
