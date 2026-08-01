from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StatusDimension:
    name: str
    percent: int
    evidence: tuple[str, ...]
    gaps: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.percent <= 100:
            raise ValueError("percent must be in [0, 100]")
        if not self.name.strip():
            raise ValueError("dimension name must not be empty")
        if not self.evidence:
            raise ValueError("dimension evidence must not be empty")
        if not self.gaps:
            raise ValueError("dimension gaps must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectStatusReport:
    schema_version: int
    prototype_scope: str
    prototype_overall_percent: int
    formal_training_overall_percent: int
    dimensions: tuple[StatusDimension, ...]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prototype_scope": self.prototype_scope,
            "prototype_overall_percent": self.prototype_overall_percent,
            "formal_training_overall_percent": self.formal_training_overall_percent,
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
            "interpretation": self.interpretation,
        }


DIMENSIONS: tuple[StatusDimension, ...] = (
    StatusDimension(
        name="engineering_quality",
        percent=92,
        evidence=(
            "pytest, Ruff, flake8 and mypy are clean locally",
            "version metadata and tracked-artifact hygiene have regression tests",
            "CLI command inventory and key JSON schemas have contract tests",
            "CI has blocking test, lint and type-check gates",
        ),
        gaps=(
            "CI matrix results still need to be observed after merge",
            "module-level non-top-level APIs are not yet snapshot-tested",
        ),
    ),
    StatusDimension(
        name="reference_architecture",
        percent=72,
        evidence=(
            "macro-block config, Delta, GQA, MoE and matched baseline are implemented",
            "streaming/chunking state contracts have unit coverage",
        ),
        gaps=(
            "global sparse attention is still a full-attention reference fallback",
            "reference operators are not scale-ready production kernels",
        ),
    ),
    StatusDimension(
        name="data_governance",
        percent=78,
        evidence=(
            "data inventory audit, decontamination and corpus split checks exist",
            "tokenizer benchmark and paired-corpus statistics are machine-readable",
        ),
        gaps=(
            "real governed corpora remain outside the repository fixture set",
            "long-term data lineage evidence must be kept current per run",
        ),
    ),
    StatusDimension(
        name="training_and_evaluation",
        percent=66,
        evidence=(
            "smoke training, checkpoint resume and validation contracts are implemented",
            "retrieval generation/scoring and paired baseline reports are covered",
        ),
        gaps=(
            "no decision-grade long-window model quality experiment exists yet",
            "current training evidence is smoke-scale only",
        ),
    ),
    StatusDimension(
        name="backend_performance",
        percent=35,
        evidence=(
            "Delta backend validation compares forward, state and gradients",
            "eager Delta and native GQA reduce some reference overhead",
        ),
        gaps=(
            "Triton/CUDA/Inductor optimized Delta, NSA and MoE kernels are not complete",
            "target-hardware positive speedup is not yet established",
        ),
    ),
    StatusDimension(
        name="zerogpu_and_operations",
        percent=70,
        evidence=(
            "ZeroGPU bundle, adapter limits and persisted evidence are documented",
            "remote quality gate path exists for bounded validation",
        ),
        gaps=(
            "operations remain tied to Space quota and external Hugging Face state",
            "latest remote quality report must be refreshed after current changes merge",
        ),
    ),
    StatusDimension(
        name="agent_runtime",
        percent=62,
        evidence=(
            "policy, tool, verifier, telemetry and conversation defaults are implemented",
            "runtime protocol validation and failure handling have tests",
        ),
        gaps=(
            "runtime is not yet bound to a trained agent-capable model",
            "credentials, permissions and long-term memory remain external by design",
        ),
    ),
    StatusDimension(
        name="documentation_and_release_readiness",
        percent=70,
        evidence=(
            "architecture, training, governance, retrieval and roadmap docs exist",
            "stage-gate and project-hygiene checks are now machine-readable",
            "top-level public API exports are protected by a snapshot test",
        ),
        gaps=(
            "API reference documentation is still incomplete",
            "release packaging and user-facing examples need hardening",
        ),
    ),
)


WEIGHTS: dict[str, int] = {
    "engineering_quality": 15,
    "reference_architecture": 15,
    "data_governance": 12,
    "training_and_evaluation": 12,
    "backend_performance": 14,
    "zerogpu_and_operations": 10,
    "agent_runtime": 10,
    "documentation_and_release_readiness": 12,
}


def build_project_status_report() -> ProjectStatusReport:
    prototype_percent = _weighted_percent(DIMENSIONS, WEIGHTS)
    return ProjectStatusReport(
        schema_version=1,
        prototype_scope="M0-M2 executable prototype, not production 28B training",
        prototype_overall_percent=prototype_percent,
        formal_training_overall_percent=38,
        dimensions=DIMENSIONS,
        interpretation=(
            "Tepid-H1 is substantially complete as an executable correctness and governance "
            "prototype. The dominant blocker to formal training readiness is optimized backend "
            "and target-hardware evidence, not basic project structure."
        ),
    )


def _weighted_percent(
    dimensions: tuple[StatusDimension, ...],
    weights: dict[str, int],
) -> int:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("status weights must have positive total weight")
    weighted = 0
    for dimension in dimensions:
        try:
            weight = weights[dimension.name]
        except KeyError as error:
            raise ValueError(f"missing weight for dimension {dimension.name!r}") from error
        weighted += dimension.percent * weight
    return round(weighted / total_weight)
