---
title: Tepid-H1 ZeroGPU Qualification
emoji: 🌌
colorFrom: indigo
colorTo: blue
sdk: gradio
python_version: 3.10.13
app_file: app.py
---

# Tepid-H1 ZeroGPU qualification

This Space runs the bounded Tepid-H1 hybrid/baseline smoke comparison inside a ZeroGPU
allocation and returns a downloadable, machine-readable report.

Select ZeroGPU in the Space hardware settings before running the app. The GPU function is
limited to five training steps and three trials, with a fixed batch size and sequence
length. The core dependency is pinned to an immutable Git commit.

The attached `himartoffice/Tepid-H1-storage` bucket is expected at `/data`. Reports are
written to `/data/reports` and survive Space rebuilds. If that volume is unavailable, the
app marks the report as an ephemeral fallback instead of claiming persistence.

## Data boundary

`paired_corpus.jsonl` is a hand-authored, non-linguistic CC0 fixture. Its exact SHA-256,
rights declaration and PII status are recorded in `data_inventory.json`. The runner
recalculates the digest and fails closed if either file is inconsistent.

The resulting losses and throughput describe only the tiny smoke configuration. They do
not establish language quality, 350M readiness or production-hardware performance.

ZeroGPU currently does not support runtime `torch.compile`, so this Space intentionally
runs the paired CUDA experiment only. Delta Inductor qualification must run on a
conventional CUDA host or be migrated to an ahead-of-time compiled artifact.
