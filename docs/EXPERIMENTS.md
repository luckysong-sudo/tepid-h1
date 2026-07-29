# Paired experiments

The paired runner verifies that Tepid-H1 and its active-parameter-matched Transformer
baseline can consume an identical data order, token budget, optimizer configuration and
finite-value training step.

The preferred CI path uses a fixed synthetic corpus whose checksum and rights metadata are
bound to an audited inventory:

```bash
tepid-h1 compare-smoke \
  --steps 2 \
  --trials 3 \
  --batch-size 1 \
  --sequence-length 8 \
  --learning-rate 0.001 \
  --seed 37 \
  --corpus configs/paired_corpus.example.jsonl \
  --inventory configs/data_inventory.example.json \
  --report artifacts/paired-governed.json
```

Until a production tokenizer is selected, the fixture stores explicit token IDs from the
smoke model's 128-token vocabulary. Each JSONL record declares `id`, `source_id`, `domain`
and `token_ids`. Loading fails closed unless:

- the inventory passes the M0 data audit;
- every record refers to one inventory source;
- the corpus file SHA-256 equals that source's declared digest;
- every record is long enough and every token ID is in vocabulary range.

The legacy random-token engineering check remains available by omitting both `--corpus`
and `--inventory`:

```bash
tepid-h1 compare-smoke \
  --steps 2 \
  --report artifacts/paired-smoke.json
```

All input batches are materialized before either model is initialized. Both models are
re-seeded identically within each trial, warmed up without gradients and trained on each
shared batch. Execution order alternates by step and trial to reduce systematic first-run
bias. The report binds the logical batch digest and records:

- equal trained-token budgets;
- loss and pre-clipping gradient norm per step;
- measured host elapsed time and token throughput;
- actual physical parameters and analytical parameter estimates;
- the active-parameter matching gap;
- per-model and paired means, sample standard deviations and normal-approximation 95% CIs;
- geometric means and log-scale 95% CIs for positive throughput ratios.

The governed fixture improves reproducibility and provenance, but it is still synthetic and
uses tiny CPU models. Its losses do not measure language quality, its normal 95% intervals
are descriptive for small trial counts, and host timings do not predict accelerator
performance. A decision-grade ablation still requires representative governed text, a
selected tokenizer, declared warmup and measurement windows, and target-hardware profiling.
