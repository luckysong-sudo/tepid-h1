#!/usr/bin/env python3
"""Example training script for Tepid-H1."""
from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tepid_h1 import (
    MixedPrecisionConfig,
    MixedPrecisionManager,
    TepidH1CausalLM,
    TepidH1Config,
    apply_gradient_checkpointing,
    log_training_step,
)


def create_model() -> TepidH1CausalLM:
    """Create model with reference configuration."""
    config = TepidH1Config.prototype()
    return TepidH1CausalLM(config)


def create_optimizer(
    model: TepidH1CausalLM,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
) -> AdamW:
    """Create AdamW optimizer with weight decay."""
    # Exclude norm parameters and bias from weight decay
    decay = set()
    no_decay = set()
    for name, param in model.named_parameters():
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.add(name)
        else:
            decay.add(name)

    param_dict = [
        {"params": [p for n, p in model.named_parameters() if n in decay], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if n in no_decay], "weight_decay": 0.0},
    ]
    return AdamW(param_dict, lr=learning_rate)


def train_step(
    model: TepidH1CausalLM,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    mp_manager: MixedPrecisionManager,
) -> dict:
    """Perform a single training step."""
    model.train()
    optimizer.zero_grad()

    with mp_manager.autocast_context():
        output = model(input_ids=input_ids.to(device), labels=labels.to(device))
        loss = output.loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return {
        "loss": loss.item(),
        "learning_rate": scheduler.get_last_lr()[0],
        "grad_norm": sum(
            p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5,
    }


def main() -> None:
    """Run training example."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = create_model()
    model.to(device)

    # Apply gradient checkpointing
    apply_gradient_checkpointing(model)

    # Setup mixed precision
    mp_manager = MixedPrecisionManager(MixedPrecisionConfig.autocast())

    # Optimizer and scheduler
    optimizer = create_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)

    # Training loop
    num_steps = 100
    batch_size = 4
    seq_len = 128

    print(f"Training for {num_steps} steps")
    for step in range(num_steps):
        # Create dummy data
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        # Training step
        metrics = train_step(model, optimizer, scheduler, input_ids, labels, device, mp_manager)

        # Log every 10 steps
        if step % 10 == 0:
            log_training_step(step, metrics, model.config)
            print(f"Step {step}: loss={metrics['loss']:.4f}, lr={metrics['learning_rate']:.2e}")

    print(f"\nTraining completed! Final loss: {metrics['loss']:.4f}")

    # Save model
    output_dir = "examples/output"
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()