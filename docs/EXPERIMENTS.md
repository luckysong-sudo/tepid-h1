# Paired smoke experiments

The paired runner verifies that Tepid-H1 and its active-parameter-matched Transformer
baseline can consume an identical data order, token budget, optimizer configuration and
finite-value training step.

```bash
tepid-h1 compare-smoke \
  --steps 2 \
  --batch-size 1 \
  --sequence-length 8 \
  --learning-rate 0.001 \
  --seed 37 \
  --report artifacts/paired-smoke.json
```

A dedicated CPU random generator materializes all input batches before either model is
initialized. Both models are re-seeded identically, warmed up without gradients and trained
on each shared batch. Execution order alternates by step to reduce systematic first-run
bias. The report binds the data SHA-256 and records:

- equal trained-token budgets;
- loss and pre-clipping gradient norm per step;
- measured host elapsed time and token throughput;
- actual physical parameters and analytical parameter estimates;
- the active-parameter matching gap.

This experiment uses random tokens and tiny models. Its losses do not measure language
quality, and host timings do not predict accelerator performance. A decision-grade ablation
must replace random tokens with a governed fixed dataset, declare warmup and measurement
windows, run repeated trials, report uncertainty, and use target-hardware profiling.
