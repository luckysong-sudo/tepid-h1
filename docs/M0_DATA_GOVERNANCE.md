# M0 data governance contract

This document defines the evidence required before a real source may enter a Tepid-H1
training mixture. The checked-in inventory is a synthetic schema fixture only; it is not a
claim that production training data has been approved.

## Admission rule

Every source must have a stable snapshot, SHA-256 digest, license identifier, rights
evidence, explicit commercial-use approval, language/domain classification, token estimate,
PII disposition and quality status. `restricted`, `prohibited`, `unknown`, PII-present and
PII-unassessed records fail closed.

Run the executable audit:

```bash
PYTHONPATH=src python -m tepid_h1.cli data-audit configs/data_inventory.example.json
```

The command exits with code `2` when any blocking finding exists, so it can gate future data
pull requests.

## Synthetic fixture

`synthetic-fixture-v1` and `synthetic-validation-v1` are separate hand-authored,
non-linguistic training and held-out validation fixtures in
`configs/paired_corpus.example.jsonl` and
`configs/validation_corpus.example.jsonl`. They exist only to exercise the schema, CI and
training control plane, contain no collected personal data and are released as CC0-1.0.
Their inventory entries bind the exact file SHA-256. The loader recalculates each digest and
rejects modified content until the inventory is intentionally updated; the training command
also rejects equal file hashes, source IDs or record IDs across the split. Replace both with
signed, snapshot-specific evidence before any real training decision.

## Repository decontamination

Decontamination is recorded at inventory scope, not inferred from source labels. A complete
record must name the comparison method, held-out benchmark sets, completion timestamp and
report location. The synthetic example compares exact hashes only; real inventories must
add normalized exact matching plus documented near-duplicate detection thresholds.

The repository provides a fail-closed comparison command:

```bash
tepid-h1 decontaminate \
  --training /path/to/training.jsonl \
  --benchmark /path/to/heldout.jsonl \
  --ngram-size 5 \
  --threshold 0.8 \
  --report artifacts/decontamination.json
```

Both inputs use `{"id":"stable-id","text":"..."}` JSONL records. Text is normalized with
Unicode NFKC, case folding and whitespace collapse. Exact normalized SHA-256 matches are
reported first; remaining records use character n-gram Jaccard similarity with an inverted
index for candidate generation. Reports contain IDs and hashes, not source text. Exit code
`3` means contamination was detected.

## Tokenizer comparison

Tokenizer selection requires one 64K, one 80K and one 96K tokenizer JSON plus a JSONL corpus
covering all three domains: `zh`, `en` and `code`.

```bash
pip install -e '.[tokenizer-eval]'
tepid-h1 tokenizer-benchmark \
  --corpus /path/to/heldout.jsonl \
  --candidate 64000=/path/to/64k.json \
  --candidate 80000=/path/to/80k.json \
  --candidate 96000=/path/to/96k.json \
  --report artifacts/tokenizer-comparison.json
```

Each corpus line has the shape `{"domain":"zh|en|code","text":"..."}`. The report binds
results to a corpus SHA-256, verifies each JSON's actual vocabulary size, records domain-level
bytes/token and characters/token, and measures both tokens/second and input UTF-8 bytes/second.
Ranking uses 70% domain-balanced compression and 30% input-byte throughput, avoiding a bias
against candidates that emit fewer tokens. The resulting selection remains provisional until
corpus licensing and representativeness are reviewed.
