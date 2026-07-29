# Reference training loop

Tepid-H1 includes a deliberately small training path for correctness checks. It is not a
distributed trainer and must not be used to claim 350M or larger-scale readiness.

## Contract

- Causal loss shifts logits left and labels right, with `-100` as the ignored label.
- Every step rejects NaN/Inf loss or gradient norm before the optimizer update.
- Gradient clipping is applied before `optimizer.step()`.
- Checkpoints are written through a temporary sibling and atomically replaced.
- A checkpoint binds model weights, optimizer state, warmup/cosine scheduler state and RNG
  state to the exact model config.
- Governed training binds the checkpoint to the corpus digest, inventory digest, source,
  batch shape, sequence length, complete AdamW recipe, gradient clipping, scheduler horizon
  and seed.
- Optional governed validation must use a different file digest, source ID and set of record
  IDs from training. Its lineage and fixed batch digest are bound into the checkpoint
  contract.
- Validation runs before and after each command under `eval()` and `torch.no_grad()`, reports
  token-weighted loss and perplexity, and restores the model's prior training mode without
  creating gradients or optimizer updates.
- Governed resume advances the corpus from the global checkpoint step and fails closed if
  either the data lineage or training contract changes, or if the scheduler step diverges
  from the checkpoint step.
- Metadata is restricted to JSON-compatible values so restricted loading remains portable.
- Loading uses PyTorch's restricted `weights_only=True` mode. Only trusted local checkpoints
  should be opened.

## Smoke run

```bash
tepid-h1 train-smoke --steps 1 --checkpoint /tmp/tepid-h1-smoke.pt
tepid-h1 train-smoke --steps 1 --checkpoint /tmp/tepid-h1-smoke.pt --resume
```

The command uses `TepidH1Config.smoke()`: an eight-layer, 32-hidden-size configuration that
still exercises Delta, local attention, global reference attention, Dense and MoE paths.
Its JSON output records loss, pre-clipping gradient norm, trained tokens and checkpoint step.

## Governed corpus run

The M2 path consumes pre-tokenized JSONL only after the referenced inventory passes audit and
the corpus digest matches its source record:

```bash
tepid-h1 train-smoke \
  --steps 1 \
  --total-steps 10 \
  --warmup-steps 2 \
  --corpus configs/paired_corpus.example.jsonl \
  --validation-corpus configs/validation_corpus.example.jsonl \
  --validation-steps 3 \
  --inventory configs/data_inventory.example.json \
  --checkpoint /tmp/tepid-h1-governed.pt \
  --report /tmp/tepid-h1-governed.json

tepid-h1 train-smoke \
  --steps 1 \
  --total-steps 10 \
  --warmup-steps 2 \
  --corpus configs/paired_corpus.example.jsonl \
  --validation-corpus configs/validation_corpus.example.jsonl \
  --validation-steps 3 \
  --inventory configs/data_inventory.example.json \
  --checkpoint /tmp/tepid-h1-governed.pt \
  --report /tmp/tepid-h1-governed-resume.json \
  --resume
```

The first command consumes corpus step 0; the resumed command starts from corpus step 1.
The report records both file digests, the inventory/source identifiers, the exact selected
batch digest, per-step learning rate, scheduler state and the half-open global step interval.
When validation is enabled, it also records the independently governed validation lineage,
fixed validation batch digest, pre/post token-weighted loss and perplexity, and both changes.
Changing the corpus, inventory or bound training recipe causes resume to stop before another
optimizer update. A requested run cannot cross the declared scheduler horizon.

## ZeroGPU evidence

GitHub Actions run
[`30455683847`](https://github.com/luckysong-sudo/tepid-h1/actions/runs/30455683847)
deployed core revision `5a6101b00612963d18400f7037ec8e7640196ee3` and
completed the full remote gate on an NVIDIA RTX PRO 6000 Blackwell Server Edition
MIG 2g.48gb:

- Ruff passed and all 54 unit tests passed (one environment-dependent skip).
- A four-step uninterrupted run and a two-step/checkpoint/resume/two-step run produced
  exactly equal model parameters with zero relative and absolute tolerance.
- The initial governed step consumed global interval `[0, 1)` and batch SHA-256
  `85ef6a997167ff6216ace9e7c441a91e1cb337cee18415c0678dccd1f53ccbdc`.
- The resumed step restored model, optimizer and RNG state, consumed `[1, 2)`, and
  produced the different batch SHA-256
  `4ad1200e38c03370aed98a06ba2af9d7e857e523f6f894f05493be2325421895`.
- The scheduler restored at completed step 1, applied warmup learning rates
  `0.0005` then `0.001`, and finished the resumed command at completed step 2.
- Both steps were bound to corpus SHA-256
  `882c60467be6da41b53ec15a650d25d5e3f612c2e32da332bb258bb36e76aa0b`
  and inventory SHA-256
  `6d05f91b63c601ab5fa7158c16db9293d02a8b5c67f9261e697b3f34221cd8d8`.
- The persistent full report is
  `/data/reports/tepid-h1-quality-2dbaf348edb14b3a97dd5a5933edc073.json`
  in `himartoffice/Tepid-H1-storage`.

This establishes the M2 control-plane path for governed data lineage, a fully bound AdamW
recipe, scheduled learning rates and exact safe resume. It does not establish useful
learning: the fixture contains only six synthetic records and the governed run performs two
optimizer steps solely to validate orchestration and invariants.
