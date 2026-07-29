# 8K and 32K retrieval evaluation

The M1 retrieval suite is deterministic infrastructure, not evidence that the current
reference model passes long-context evaluation. The global reference attention limit still
prevents a full 32K model run.

## Generate

```bash
tepid-h1 retrieval-generate \
  --prompts artifacts/retrieval-prompts.jsonl \
  --answers artifacts/retrieval-answers.jsonl \
  --length 8192 \
  --length 32768 \
  --position 0.1 \
  --position 0.5 \
  --position 0.9 \
  --seed 41
```

Lengths use the declared `whitespace-v1` reference tokenizer, making every prompt exactly
8,192 or 32,768 tokens without depending on an unselected production tokenizer. Each case
contains a random key/value needle and a final key query. Early, middle and late insertion
positions are covered.

Prompts and answer keys are written separately. Prompt records never contain an answer
field. Once a production tokenizer is selected, a new version of the suite must bind lengths
to that tokenizer and retain the current suite as a stable regression fixture.

## Score

Model predictions use JSONL records shaped as `{"case_id":"...","answer":"..."}`.

```bash
tepid-h1 retrieval-score \
  --answers artifacts/retrieval-answers.jsonl \
  --predictions artifacts/model-predictions.jsonl \
  --minimum-accuracy 1.0 \
  --report artifacts/retrieval-report.json
```

Scoring requires full case coverage and reports exact-copy accuracy by target length and
needle position. Unknown or duplicate case IDs are rejected. Missing and incorrect answers
are recorded without copying expected answers into the report. Exit code `4` means the gate
failed.
