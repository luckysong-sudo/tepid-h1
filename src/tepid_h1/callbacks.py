"""Training callbacks and monitoring for Tepid-H1."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class TrainingCallback:
    """Callback invoked at key training events."""

    on_step: Callable[[int, dict[str, Any]], None] | None = None
    on_epoch: Callable[[int, dict[str, Any]], None] | None = None
    on_checkpoint: Callable[[int, dict[str, Any]], None] | None = None
    on_error: Callable[[Exception, dict[str, Any]], None] | None = None


@dataclass
class TrainingMetricsBuffer:
    """Thread-safe metrics buffer for rolling statistics."""

    window_size: int = 100
    _losses: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _grad_norms: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _throughputs: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record(
        self,
        loss: float,
        gradient_norm: float,
        throughput: float,
    ) -> None:
        self._losses.append(loss)
        self._grad_norms.append(gradient_norm)
        self._throughputs.append(throughput)
        self._timestamps.append(time.perf_counter())

    def summary(self) -> dict[str, Any]:
        if not self._losses:
            return {"samples": 0}
        return {
            "samples": len(self._losses),
            "loss_mean": sum(self._losses) / len(self._losses),
            "loss_std": self._std(self._losses),
            "loss_min": min(self._losses),
            "loss_max": max(self._losses),
            "grad_norm_mean": sum(self._grad_norms) / len(self._grad_norms),
            "throughput_mean": sum(self._throughputs) / len(self._throughputs),
        }

    @staticmethod
    def _std(values: deque[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5

    def is_stable(self, *, tolerance: float = 0.01, window: int | None = None) -> bool:
        if len(self._losses) < 10:
            return False
        check_window = window or min(len(self._losses), 50)
        recent = list(self._losses)[-check_window:]
        return (max(recent) - min(recent)) / (abs(sum(recent) / len(recent)) + 1e-12) < tolerance


class EarlyStopper:
    """Early stopping based on validation loss improvement."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ) -> None:
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float | None = None
        self._counter = 0
        self._should_stop = False

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def best_value(self) -> float | None:
        return self._best

    def update(self, value: float) -> bool:
        if self.mode == "min":
            improved = self._best is None or value < self._best - self.min_delta
        else:
            improved = self._best is None or value > self._best + self.min_delta

        if improved:
            self._best = value
            self._counter = 0
            return True
        self._counter += 1
        if self._counter >= self.patience:
            self._should_stop = True
        return False


class LossTracker:
    """Track loss across training steps for convergence analysis."""

    def __init__(self) -> None:
        self._losses: list[float] = []
        self._steps: list[int] = []
        self._start_time: float = time.perf_counter()

    def record(self, step: int, loss: float) -> None:
        self._losses.append(loss)
        self._steps.append(step)

    @property
    def losses(self) -> list[float]:
        return list(self._losses)

    @property
    def steps(self) -> list[int]:
        return list(self._steps)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start_time

    def convergence_rate(self, window: int = 10) -> float | None:
        if len(self._losses) < window:
            return None
        recent = self._losses[-window:]
        diffs = [recent[i] - recent[i + 1] for i in range(len(recent) - 1)]
        return sum(diffs) / len(diffs)

    def report(self) -> dict[str, Any]:
        if not self._losses:
            return {"status": "no_data"}
        return {
            "total_steps": len(self._losses),
            "current_loss": self._losses[-1],
            "initial_loss": self._losses[0],
            "total_reduction": self._losses[0] - self._losses[-1],
            "reduction_percent": (
                (self._losses[0] - self._losses[-1]) / (self._losses[0] + 1e-12) * 100
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "convergence_rate": self.convergence_rate(),
        }


class TrainingRunner:
    """Enhanced training loop with callbacks and monitoring."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        callbacks: list[TrainingCallback] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.metrics = TrainingMetricsBuffer()
        self.loss_tracker = LossTracker()
        self.early_stopper = EarlyStopper()
        self._step_count = 0

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        max_gradient_norm: float = 1.0,
    ) -> dict[str, Any]:
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(input_ids, labels=labels)
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        if not torch.isfinite(output.loss):
            raise RuntimeError("training loss is NaN or Inf")

        output.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_gradient_norm
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("gradient norm is NaN or Inf")
        self.optimizer.step()
        self._step_count += 1

        elapsed = time.perf_counter() - self.loss_tracker._start_time
        throughput = self._step_count / (elapsed + 1e-12)

        metrics = {
            "step": self._step_count,
            "loss": float(output.loss.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "throughput": throughput,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }
        self.metrics.record(metrics["loss"], metrics["gradient_norm"], metrics["throughput"])
        self.loss_tracker.record(self._step_count, metrics["loss"])

        for callback in self.callbacks:
            if callback.on_step is not None:
                try:
                    callback.on_step(self._step_count, metrics)
                except Exception as exc:
                    for cb in self.callbacks:
                        if cb.on_error is not None:
                            try:
                                cb.on_error(exc, metrics)
                            except Exception:
                                pass
        return metrics

    def train_epoch(
        self,
        batches: list[torch.Tensor],
        labels_batches: list[torch.Tensor | None] | None = None,
        *,
        max_gradient_norm: float = 1.0,
    ) -> dict[str, Any]:
        if labels_batches is None:
            labels_batches = [None] * len(batches)
        if len(labels_batches) != len(batches):
            raise ValueError("labels_batches must match batches length")

        epoch_start = time.perf_counter()
        epoch_losses: list[float] = []

        for step_idx, (input_ids, labels) in enumerate(zip(batches, labels_batches, strict=True)):
            try:
                metrics = self.train_step(
                    input_ids,
                    labels,
                    max_gradient_norm=max_gradient_norm,
                )
                epoch_losses.append(metrics["loss"])
            except Exception as exc:
                for callback in self.callbacks:
                    if callback.on_error is not None:
                        try:
                            callback.on_error(exc, {"step": step_idx})
                        except Exception:
                            pass
                raise

        elapsed = time.perf_counter() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        epoch_metrics = {
            "epoch": self._step_count,
            "avg_loss": avg_loss,
            "elapsed_seconds": elapsed,
            "steps_per_second": len(batches) / (elapsed + 1e-12),
        }
        for callback in self.callbacks:
            if callback.on_epoch is not None:
                try:
                    callback.on_epoch(self._step_count, epoch_metrics)
                except Exception:
                    pass
        return epoch_metrics

    def checkpoint(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": self._step_count,
            "metrics_summary": self.metrics.summary(),
            "loss_tracker": {
                "losses": self.loss_tracker.losses,
                "steps": self.loss_tracker.steps,
            },
        }
        torch.save(state, path)
        for callback in self.callbacks:
            if callback.on_checkpoint is not None:
                try:
                    callback.on_checkpoint(self._step_count, {"path": str(path)})
                except Exception:
                    pass
        return state