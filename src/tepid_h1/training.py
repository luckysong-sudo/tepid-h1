from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from .modeling import TepidH1CausalLM, TransformerBaselineCausalLM

TrainableCausalLM = TepidH1CausalLM | TransformerBaselineCausalLM


class NonFiniteTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainStepMetrics:
    loss: float
    gradient_norm: float
    trained_tokens: int
    learning_rate: float


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    perplexity: float
    evaluated_tokens: int
    batches: int


@dataclass(frozen=True)
class CheckpointState:
    step: int
    metadata: Mapping[str, Any]


@dataclass
class WarmupCosineScheduler:
    optimizer: torch.optim.Optimizer
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.1
    completed_steps: int = field(init=False, default=0)
    base_lrs: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.warmup_steps, int) or isinstance(self.warmup_steps, bool):
            raise TypeError("warmup_steps must be an integer")
        if not isinstance(self.total_steps, int) or isinstance(self.total_steps, bool):
            raise TypeError("total_steps must be an integer")
        if not 0 <= self.warmup_steps <= self.total_steps:
            raise ValueError("warmup_steps must be between zero and total_steps")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between zero and one")
        self.base_lrs = tuple(float(group["lr"]) for group in self.optimizer.param_groups)
        self._apply_learning_rates()

    def step(self) -> None:
        if self.completed_steps >= self.total_steps:
            raise ValueError("learning-rate schedule is already complete")
        self.completed_steps += 1
        self._apply_learning_rates()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "completed_steps": self.completed_steps,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": 1,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
        }
        actual = {key: payload.get(key) for key in expected}
        if actual != expected:
            raise ValueError("checkpoint learning-rate schedule does not match the current run")
        completed_steps = payload.get("completed_steps")
        if (
            not isinstance(completed_steps, int)
            or isinstance(completed_steps, bool)
            or not 0 <= completed_steps <= self.total_steps
        ):
            raise ValueError("checkpoint scheduler step is invalid")
        self.completed_steps = completed_steps
        self._apply_learning_rates()

    def _apply_learning_rates(self) -> None:
        factor = self._factor(self.completed_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def _factor(self, update_index: int) -> float:
        if update_index >= self.total_steps:
            return self.min_lr_ratio
        if self.warmup_steps and update_index < self.warmup_steps:
            return (update_index + 1) / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 1:
            return self.min_lr_ratio
        progress = (update_index - self.warmup_steps) / (decay_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


def validate_resume_contract(
    checkpoint_metadata: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> None:
    saved_contract = checkpoint_metadata.get("training_contract")
    if not isinstance(saved_contract, Mapping):
        raise TypeError("checkpoint does not contain a training contract")
    if dict(saved_contract) != dict(expected_contract):
        raise ValueError("checkpoint training contract does not match the current run")


def causal_lm_train_step(
    model: TrainableCausalLM,
    input_ids: Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    labels: Tensor | None = None,
    max_gradient_norm: float = 1.0,
) -> TrainStepMetrics:
    if max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be positive")
    learning_rate = float(optimizer.param_groups[0]["lr"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    targets = input_ids if labels is None else labels
    trained_tokens = _count_supervised_target_tokens(input_ids, targets, context="training")
    output = model(input_ids, labels=targets)
    if output.loss is None:
        raise AssertionError("model did not return a training loss")
    if not torch.isfinite(output.loss):
        raise NonFiniteTrainingError("training loss is NaN or Inf")

    output.loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    if not torch.isfinite(gradient_norm):
        optimizer.zero_grad(set_to_none=True)
        raise NonFiniteTrainingError("gradient norm is NaN or Inf")
    optimizer.step()
    return TrainStepMetrics(
        loss=float(output.loss.detach()),
        gradient_norm=float(gradient_norm.detach()),
        trained_tokens=trained_tokens,
        learning_rate=learning_rate,
    )


def evaluate_causal_lm(
    model: TrainableCausalLM,
    batches: tuple[Tensor, ...],
    *,
    labels_batches: tuple[Tensor | None, ...] | None = None,
) -> EvaluationMetrics:
    if not batches:
        raise ValueError("evaluation requires at least one batch")
    if labels_batches is None:
        labels_batches = tuple(None for _ in batches)
    if len(labels_batches) != len(batches):
        raise ValueError("labels_batches must match batches length")
    was_training = model.training
    weighted_loss = 0.0
    evaluated_tokens = 0
    try:
        model.eval()
        with torch.no_grad():
            for input_ids, labels in zip(batches, labels_batches, strict=True):
                targets = input_ids if labels is None else labels
                batch_tokens = _count_supervised_target_tokens(
                    input_ids, targets, context="evaluation"
                )
                output = model(input_ids, labels=targets)
                if output.loss is None or not torch.isfinite(output.loss):
                    raise NonFiniteTrainingError("evaluation loss is NaN or Inf")
                weighted_loss += float(output.loss) * batch_tokens
                evaluated_tokens += batch_tokens
    finally:
        model.train(was_training)
    mean_loss = weighted_loss / evaluated_tokens
    perplexity = math.exp(mean_loss)
    if not math.isfinite(perplexity):
        raise NonFiniteTrainingError("evaluation perplexity is NaN or Inf")
    return EvaluationMetrics(
        loss=mean_loss,
        perplexity=perplexity,
        evaluated_tokens=evaluated_tokens,
        batches=len(batches),
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: TrainableCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler | None = None,
    step: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    if scheduler is not None and scheduler.completed_steps != step:
        raise ValueError("checkpoint scheduler step must match checkpoint step")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = dict(metadata or {})
    try:
        json.dumps(checkpoint_metadata)
    except (TypeError, ValueError) as error:
        raise TypeError("checkpoint metadata must contain only JSON-compatible values") from error
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    payload = {
        "schema_version": 2,
        "config": _serialized_model_config(model),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "step": step,
        "metadata": checkpoint_metadata,
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    model: TrainableCausalLM,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: WarmupCosineScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("config") != _serialized_model_config(model):
        raise ValueError("checkpoint model config does not match the target model")
    step = payload.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a mapping")
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    scheduler_state = payload.get("scheduler_state")
    rng_state = payload.get("rng_state")
    if not isinstance(model_state, Mapping):
        raise TypeError("checkpoint model_state must be a mapping")
    if optimizer is not None and not isinstance(optimizer_state, dict):
        raise TypeError("checkpoint optimizer_state must be a mapping")
    if not isinstance(rng_state, Tensor):
        raise TypeError("checkpoint rng_state must be a tensor")
    if scheduler is not None:
        _validate_checkpoint_scheduler_state(scheduler_state, scheduler, step)

    model.load_state_dict(model_state, strict=True)
    if optimizer is not None:
        if not isinstance(optimizer_state, dict):
            raise TypeError("checkpoint optimizer_state must be a mapping")
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(cast(Mapping[str, Any], scheduler_state))
    torch.set_rng_state(rng_state.cpu())
    cuda_rng_states = payload.get("cuda_rng_states", [])
    if torch.cuda.is_available() and cuda_rng_states:
        torch.cuda.set_rng_state_all(cuda_rng_states)
    return CheckpointState(step=step, metadata=metadata)


def _serialized_model_config(model: TrainableCausalLM) -> dict[str, Any]:
    if isinstance(model, TransformerBaselineCausalLM):
        return model.baseline_config.to_dict()
    return model.config.to_dict()


def _validate_checkpoint_scheduler_state(
    scheduler_state: Any,
    scheduler: WarmupCosineScheduler,
    step: int,
) -> None:
    if not isinstance(scheduler_state, Mapping):
        raise TypeError("checkpoint scheduler_state must be a mapping")
    expected = {
        "schema_version": 1,
        "warmup_steps": scheduler.warmup_steps,
        "total_steps": scheduler.total_steps,
        "min_lr_ratio": scheduler.min_lr_ratio,
        "base_lrs": list(scheduler.base_lrs),
    }
    actual = {key: scheduler_state.get(key) for key in expected}
    if actual != expected:
        raise ValueError("checkpoint learning-rate schedule does not match the current run")
    completed_steps = scheduler_state.get("completed_steps")
    if (
        not isinstance(completed_steps, int)
        or isinstance(completed_steps, bool)
        or not 0 <= completed_steps <= scheduler.total_steps
    ):
        raise ValueError("checkpoint scheduler step is invalid")
    if completed_steps != step:
        raise ValueError("checkpoint scheduler step does not match checkpoint step")


def _count_supervised_target_tokens(
    input_ids: Tensor,
    labels: Tensor,
    *,
    context: str,
) -> int:
    if input_ids.ndim != 2:
        raise ValueError(f"{context} input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] <= 0:
        raise ValueError(f"{context} input_ids batch size must be positive")
    if input_ids.dtype != torch.long:
        raise TypeError(f"{context} input_ids must use torch.long dtype")
    if labels.shape != input_ids.shape:
        raise ValueError(f"{context} labels must have the same shape as input_ids")
    if labels.dtype != torch.long:
        raise TypeError(f"{context} labels must use torch.long dtype")
    if labels.shape[1] < 2:
        raise ValueError(f"{context} labels must include at least one target token")
    target_count = int((labels[:, 1:] != -100).sum().item())
    if target_count <= 0:
        raise ValueError(f"{context} labels must include at least one target token")
    return target_count
