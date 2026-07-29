# Reference training loop

Tepid-H1 includes a deliberately small training path for correctness checks. It is not a
distributed trainer and must not be used to claim 350M or larger-scale readiness.

## Contract

- Causal loss shifts logits left and labels right, with `-100` as the ignored label.
- Every step rejects NaN/Inf loss or gradient norm before the optimizer update.
- Gradient clipping is applied before `optimizer.step()`.
- Checkpoints are written through a temporary sibling and atomically replaced.
- A checkpoint binds model weights, optimizer state and RNG state to the exact model config.
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
