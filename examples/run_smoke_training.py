#!/usr/bin/env python3
"""
Tepid-H1 Smoke Training Example

This script demonstrates running a tiny deterministic causal-LM training loop
using the Tepid-H1 model architecture.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tepid-H1 smoke training")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("smoke_output.json"))
    return run_training(parser.parse_args())


def run_training(args: argparse.Namespace) -> int:
    """Run a small training loop with deterministic tokens."""
    from src.tepid_h1.config import TepidH1Config
    from src.tepid_h1.modeling import TepidH1CausalLM
    from src.tepid_h1.training import (
        WarmupCosineScheduler,
        causal_lm_train_step,
        evaluate_causal_lm,
    )

    # Initialize config and model
    config = TepidH1Config.smoke()
    model = TepidH1CausalLM(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=2,
        total_steps=args.steps,
    )

    # Generate deterministic input
    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, config.vocab_size, (args.batch_size, args.sequence_length)
    )

    metrics = []
    for step in range(args.steps):
        step_metrics = causal_lm_train_step(model, input_ids, optimizer)
        scheduler.step()
        metrics.append({
            "step": step,
            "loss": step_metrics.loss,
            "gradient_norm": step_metrics.gradient_norm,
            "learning_rate": step_metrics.learning_rate,
        })
        print(f"Step {step}: loss={step_metrics.loss:.4f}, lr={step_metrics.learning_rate:.6f}")

    # Evaluation
    eval_metrics = evaluate_causal_lm(model, (input_ids,))

    result = {
        "final_loss": metrics[-1]["loss"] if metrics else None,
        "evaluation_loss": eval_metrics.loss,
        "evaluation_perplexity": eval_metrics.perplexity,
        "steps_completed": len(metrics),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())