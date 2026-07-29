# Active-parameter-matched Transformer baseline

M1 requires a standard Transformer control for hybrid architecture comparisons. The
reference baseline uses the same vocabulary, hidden size, layer count, GQA head geometry,
normalization, embedding tying and initialization as Tepid-H1, while replacing every
sequence mixer with full causal GQA and every channel mixer with Dense SwiGLU.

## Matching rule

```bash
tepid-h1 baseline-report --variant reference
```

The tool estimates parameters touched per token:

- Delta: input and output projection parameters;
- attention: Q/K/V/output projection parameters;
- Dense SwiGLU: all gate/up/down parameters;
- MoE: router, shared expert and Top-K selected expert parameters;
- common: embeddings, norms and residual scales.

It then solves the baseline FFN intermediate width that minimizes the active-parameter gap.
Physical MoE parameters are reported separately.

This is a transparent proxy, not a measured compute match. It does not include activation
shape, attention quadratic cost, recurrent state traffic, kernel utilization,
communication, latency or memory bandwidth. A model-quality claim requires the same
tokenizer, data order, optimizer, token budget and evaluation suite, followed by measured
FLOPs and target-hardware results.

## Correctness boundary

The baseline supports the same attention KV state contract and causal loss as the reference
hybrid model. It also uses the same finite-value-checked training-step function, while
checkpoints bind the derived baseline FFN width in addition to the shared model config. Unit
tests require full-pass and chunked logits to agree before the baseline may be used in
training comparisons.
