"""Training correctness and governance improvements for Tepid-H1.

This module adds machine-readable evidence records that document the
training contract gaps, masked-label handling and supervised-target
validation that have been added after the initial smoke-train baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingImprovement:
    """A single training improvement with machine-readable evidence."""

    id: str
    category: str
    description: str
    contract_added: bool = True
    test_coverage: bool = True
    gap_remaining: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRAINING_IMPROVEMENTS: tuple[TrainingImprovement, ...] = (
    TrainingImprovement(
        id="training-target-validation",
        category="training",
        description=(
            "training rejects batches without supervised target tokens before forward"
        ),
    ),
    TrainingImprovement(
        id="training-eval-label-shape-validation",
        category="training",
        description=(
            "training and evaluation validate supervised target shapes before forward"
        ),
    ),
    TrainingImprovement(
        id="training-eval-batch-dtype-cardinality-validation",
        category="training",
        description=(
            "training and evaluation reject invalid input batch dtype and cardinality "
            "before forward"
        ),
    ),
    TrainingImprovement(
        id="callback-training-empty-epochs",
        category="callbacks",
        description=(
            "callback training runner rejects empty epochs and invalid clipping controls"
        ),
    ),
    TrainingImprovement(
        id="gradient-checkpointing-static-selection",
        category="gradient_checkpointing",
        description=(
            "gradient checkpointing selects layers statically and rejects invalid controls"
        ),
    ),
    TrainingImprovement(
        id="mixed-precision-tensor-dtype-safety",
        category="mixed_precision",
        description=(
            "mixed precision preserves token tensor dtype and restores runtime state safely"
        ),
    ),
    TrainingImprovement(
        id="checkpoint-save-invalid-step-validation",
        category="training",
        description=(
            "checkpoint saving rejects invalid step types and scheduler-step mismatches"
        ),
    ),
    TrainingImprovement(
        id="checkpoint-load-scheduler-mismatch",
        category="training",
        description=(
            "checkpoint loading rejects scheduler mismatches before mutating model state"
        ),
    ),
    TrainingImprovement(
        id="checkpoint-load-metadata-rng-validation",
        category="training",
        description=(
            "checkpoint loading validates metadata and CPU/CUDA RNG payloads before state restore"
        ),
    ),
    TrainingImprovement(
        id="paired-smoke-invalid-controls",
        category="experiments",
        description=(
            "paired smoke configuration rejects ambiguous and non-finite training controls"
        ),
    ),
)


def list_training_improvements() -> tuple[TrainingImprovement, ...]:
    """Return all registered training improvement records."""
    return TRAINING_IMPROVEMENTS


def get_training_improvement_ids() -> tuple[str, ...]:
    """Return the machine-readable improvement IDs."""
    return tuple(imp.id for imp in TRAINING_IMPROVEMENTS)


def count_training_improvements() -> int:
    """Return the number of registered training improvements."""
    return len(TRAINING_IMPROVEMENTS)


def filter_training_improvements(
    improvements: tuple[TrainingImprovement, ...] | None = None,
    *,
    category: str | None = None,
) -> tuple[TrainingImprovement, ...]:
    """Filter training improvements by category.

    Args:
        improvements: Optional tuple of improvements to filter.
            Defaults to all registered improvements.
        category: Optional category name to filter by.
            If None, all improvements are returned.

    Returns:
        Filtered tuple of training improvements.
    """
    if improvements is None:
        improvements = TRAINING_IMPROVEMENTS
    if category is None:
        return improvements
    return tuple(imp for imp in improvements if imp.category == category)
