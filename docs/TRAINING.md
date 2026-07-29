# Reference training loop

Tepid-H1 includes a deliberately small training path for correctness checks. It is not a
distributed trainer and must not be used to claim 350M or larger-scale readiness.

## Contract

- Causal loss shifts logits left and labels right, with `-100` as the ignored label.
- Every step rejects NaN/Inf loss or gradient norm before the optimizer update.
- Gradient clipping is applied before `optimizer.step()`.
- Checkpoints are written through a temporary sibling and atomically replaced.
- A checkpoint binds model weights, optimizer state and RNG state to the exact model config.
- Governed training binds the checkpoint to the corpus digest, inventory digest, source,
  batch shape, sequence length, learning rate and seed.
- Governed resume advances the corpus from the global checkpoint step and fails closed if
  either the data lineage or training contract changes.
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
  --corpus configs/paired_corpus.example.jsonl \
  --inventory configs/data_inventory.example.json \
  --checkpoint /tmp/tepid-h1-governed.pt \
  --report /tmp/tepid-h1-governed.json

tepid-h1 train-smoke \
  --steps 1 \
  --corpus configs/paired_corpus.example.jsonl \
  --inventory configs/data_inventory.example.json \
  --checkpoint /tmp/tepid-h1-governed.pt \
  --report /tmp/tepid-h1-governed-resume.json \
  --resume
```

The first command consumes corpus step 0; the resumed command starts from corpus step 1.
The report records both file digests, the inventory/source identifiers, the exact selected
batch digest and the half-open global step interval. Changing the corpus, inventory or bound
training recipe causes resume to stop before another optimizer update.

## ZeroGPU evidence

GitHub Actions run
[`30454160329`](https://github.com/luckysong-sudo/tepid-h1/actions/runs/30454160329)
deployed core revision `df0a70cf36ea2c6ac28865327d47bf051509c4d6` and
completed the full remote gate on an NVIDIA RTX PRO 6000 Blackwell Server Edition
MIG 2g.48gb:

- Ruff passed and all 53 unit tests passed (one environment-dependent skip).
- The initial governed step consumed global interval `[0, 1)` and batch SHA-256
  `85ef6a997167ff6216ace9e7c441a91e1cb337cee18415c0678dccd1f53ccbdc`.
- The resumed step restored model, optimizer and RNG state, consumed `[1, 2)`, and
  produced the different batch SHA-256
  `4ad1200e38c03370aed98a06ba2af9d7e857e523f6f894f05493be2325421895`.
- Both steps were bound to corpus SHA-256
  `882c60467be6da41b53ec15a650d25d5e3f612c2e32da332bb258bb36e76aa0b`
  and inventory SHA-256
  `6d05f91b63c601ab5fa7158c16db9293d02a8b5c67f9261e697b3f34221cd8d8`.
- The persistent full report is
  `/data/reports/tepid-h1-quality-035fa57d3b4146abb682764f7e7f6193.json`
  in `himartoffice/Tepid-H1-storage`.

This establishes the M2 control-plane path for governed data lineage and safe resume. It
does not establish useful learning: the fixture contains only six synthetic records and the
run performs two optimizer steps solely to validate orchestration and invariants.
