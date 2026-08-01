"""Metrics collection and reporting utilities for training loops."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricBucket:
    """Sliding window metric collector with statistics."""

    window_size: int = 100
    _values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > self.window_size:
            self._values.pop(0)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def min_value(self) -> float:
        if not self._values:
            return 0.0
        return min(self._values)

    @property
    def max_value(self) -> float:
        if not self._values:
            return 0.0
        return max(self._values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": round(self.mean, 6),
            "min": round(self.min_value, 6),
            "max": round(self.max_value, 6),
        }


@dataclass
class TrainingMetrics:
    """Collects and reports training metrics."""

    loss: MetricBucket = field(default_factory=MetricBucket)
    gradient_norm: MetricBucket = field(default_factory=MetricBucket)
    learning_rate: MetricBucket = field(default_factory=MetricBucket)
    throughput: MetricBucket = field(default_factory=MetricBucket)

    def record_step(
        self,
        loss: float,
        gradient_norm: float,
        learning_rate: float,
        throughput: float | None = None,
    ) -> None:
        self.loss.add(loss)
        self.gradient_norm.add(gradient_norm)
        self.learning_rate.add(learning_rate)
        if throughput is not None:
            self.throughput.add(throughput)

    def summary(self) -> dict[str, Any]:
        return {
            "loss": self.loss.to_dict(),
            "gradient_norm": self.gradient_norm.to_dict(),
            "learning_rate": self.learning_rate.to_dict(),
            "throughput": self.throughput.to_dict() if self.throughput.count > 0 else None,
        }