from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from .config import TepidH1Config
from .modeling import (
    TepidH1CausalLM,
    TransformerBaselineCausalLM,
    TransformerBaselineConfig,
    baseline_parameter_estimate,
    hybrid_parameter_estimate,
)
from .training import TrainableCausalLM, causal_lm_train_step


@dataclass(frozen=True)
class PairedExperimentConfig:
    steps: int = 2
    batch_size: int = 1
    sequence_length: int = 8
    learning_rate: float = 1e-3
    max_gradient_norm: float = 1.0
    seed: int = 37

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if not 2 <= self.sequence_length <= 64:
            raise ValueError("sequence_length must be between 2 and 64")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive")


def run_paired_smoke(config: PairedExperimentConfig) -> dict[str, Any]:
    model_config = TepidH1Config.smoke()
    baseline_config = TransformerBaselineConfig.active_parameter_matched(model_config)
    batches = _generate_batches(config, model_config.vocab_size)

    torch.manual_seed(config.seed)
    hybrid = TepidH1CausalLM(model_config)
    torch.manual_seed(config.seed)
    baseline = TransformerBaselineCausalLM(baseline_config)
    optimizers = {
        "hybrid": torch.optim.AdamW(hybrid.parameters(), lr=config.learning_rate),
        "baseline": torch.optim.AdamW(baseline.parameters(), lr=config.learning_rate),
    }
    models: dict[str, TrainableCausalLM] = {"hybrid": hybrid, "baseline": baseline}
    _warm_up(models, batches[0])

    measurements: dict[str, list[dict[str, float | int]]] = {
        "hybrid": [],
        "baseline": [],
    }
    execution_order: list[list[str]] = []
    for step, input_ids in enumerate(batches):
        order = ["hybrid", "baseline"] if step % 2 == 0 else ["baseline", "hybrid"]
        execution_order.append(order)
        for name in order:
            started = time.perf_counter()
            metrics = causal_lm_train_step(
                models[name],
                input_ids,
                optimizers[name],
                max_gradient_norm=config.max_gradient_norm,
            )
            elapsed = time.perf_counter() - started
            measurements[name].append({**asdict(metrics), "elapsed_seconds": elapsed})

    hybrid_estimate = hybrid_parameter_estimate(model_config)
    baseline_estimate = baseline_parameter_estimate(baseline_config)
    return {
        "schema_version": 1,
        "experiment": "paired_random_token_smoke",
        "interpretation": (
            "Engineering comparability smoke only; random-token loss and host timing "
            "are not model-quality or production-performance evidence."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "config": asdict(config),
        "data": {
            "sha256": _batch_digest(batches),
            "batches": len(batches),
            "tokens_per_model": config.steps * config.batch_size * (config.sequence_length - 1),
        },
        "execution_order": execution_order,
        "hybrid": _summarize(
            measurements["hybrid"],
            actual_parameters=_parameter_count(hybrid),
            estimated_active_parameters=hybrid_estimate["active_parameters"],
            estimated_physical_parameters=hybrid_estimate["physical_parameters"],
        ),
        "baseline": _summarize(
            measurements["baseline"],
            actual_parameters=_parameter_count(baseline),
            estimated_active_parameters=int(baseline_estimate["active_parameters"]),
            estimated_physical_parameters=int(baseline_estimate["physical_parameters"]),
        ),
        "matching": {
            "basis": "per-token active-parameter proxy",
            "baseline_intermediate_size": baseline_config.intermediate_size,
            "active_parameter_gap_percent": baseline_estimate["active_parameter_gap_percent"],
        },
    }


def _generate_batches(config: PairedExperimentConfig, vocab_size: int) -> tuple[Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 1)
    return tuple(
        torch.randint(
            0,
            vocab_size,
            (config.batch_size, config.sequence_length),
            generator=generator,
        )
        for _ in range(config.steps)
    )


def _warm_up(models: dict[str, TrainableCausalLM], input_ids: Tensor) -> None:
    warmup_ids = input_ids[:, : min(input_ids.shape[1], 4)]
    with torch.no_grad():
        for model in models.values():
            model.eval()
            model(warmup_ids)


def _batch_digest(batches: tuple[Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _parameter_count(model: TrainableCausalLM) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _summarize(
    measurements: list[dict[str, float | int]],
    *,
    actual_parameters: int,
    estimated_active_parameters: int,
    estimated_physical_parameters: int,
) -> dict[str, Any]:
    elapsed = sum(float(item["elapsed_seconds"]) for item in measurements)
    trained_tokens = sum(int(item["trained_tokens"]) for item in measurements)
    losses = [float(item["loss"]) for item in measurements]
    return {
        "actual_parameters": actual_parameters,
        "estimated_active_parameters": estimated_active_parameters,
        "estimated_physical_parameters": estimated_physical_parameters,
        "parameter_estimate_matches_actual": actual_parameters == estimated_physical_parameters,
        "trained_tokens": trained_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": trained_tokens / elapsed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_change": losses[-1] - losses[0],
        "steps": measurements,
    }
