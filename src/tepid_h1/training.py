from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class CheckpointState:
    step: int
    metadata: Mapping[str, Any]


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
    model.train()
    optimizer.zero_grad(set_to_none=True)
    targets = input_ids if labels is None else labels
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
    trained_tokens = int((targets[:, 1:] != -100).sum().item())
    return TrainStepMetrics(
        loss=float(output.loss.detach()),
        gradient_norm=float(gradient_norm.detach()),
        trained_tokens=trained_tokens,
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: TrainableCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
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
        "schema_version": 1,
        "config": _serialized_model_config(model),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
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
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
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
    rng_state = payload.get("rng_state")
    if not isinstance(model_state, Mapping):
        raise TypeError("checkpoint model_state must be a mapping")
    if optimizer is not None and not isinstance(optimizer_state, dict):
        raise TypeError("checkpoint optimizer_state must be a mapping")
    if not isinstance(rng_state, Tensor):
        raise TypeError("checkpoint rng_state must be a tensor")

    model.load_state_dict(model_state, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(rng_state.cpu())
    cuda_rng_states = payload.get("cuda_rng_states", [])
    if torch.cuda.is_available() and cuda_rng_states:
        torch.cuda.set_rng_state_all(cuda_rng_states)
    return CheckpointState(step=step, metadata=metadata)


def _serialized_model_config(model: TrainableCausalLM) -> dict[str, Any]:
    if isinstance(model, TransformerBaselineCausalLM):
        return model.baseline_config.to_dict()
    return model.config.to_dict()
