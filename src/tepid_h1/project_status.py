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
            "GPU preflight readiness exit codes are covered by CLI contract tests",
            "top-level and subpackage public APIs have snapshot tests",
            "CI has blocking test, lint and type-check gates",
        ),
        gaps=("CI matrix results still need to be observed after merge",),
    ),
    StatusDimension(
        name="reference_architecture",
        percent=79,
        evidence=(
            "macro-block config, Delta, GQA, MoE and matched baseline are implemented",
            "streaming/chunking state contracts have unit coverage",
            "global sparse attention now uses deterministic local-window plus "
            "global-anchor sparsity",
            "model recurrent state inputs fail closed when Delta or attention state "
            "counts do not match the layer plan",
            "model input token IDs fail closed on dtype, sequence length and vocab range",
        ),
        gaps=(
            "reference sparse attention is not yet a production fused kernel",
            "Delta and MoE reference operators are not scale-ready production kernels",
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
        percent=88,
        evidence=(
            "smoke training, checkpoint resume and validation contracts are implemented",
            "retrieval generation/scoring and paired baseline reports are covered",
            "MoE router load-balancing auxiliary loss is integrated into CausalLM training",
            "streaming train/eval calls reject incomplete recurrent state tuples",
            "causal-LM loss validates label dtype, ignore index and vocab range before training",
            "autoregressive inference respects explicit greedy versus sampling mode",
            "repetition penalty uses generated-token history during autoregressive decoding",
            "top-k and nucleus sampling filters handle vocabulary-index boundaries correctly",
            "autoregressive sampling suppresses pad tokens while preserving eos stop tokens",
            "batched generation masks rows that have already emitted eos",
            "generation validates and applies configured device and dtype backends",
            "generation samples the first new token from prompt prefill logits",
            "nucleus sampling keeps the minimum token set that crosses the top-p threshold",
            "sampling applies temperature before top-k and nucleus filtering",
            "generation validates special token IDs against the model vocabulary",
            "generation configuration rejects ambiguous control-value types",
            "evaluation supports masked labels and weights loss by valid target tokens",
            "training rejects batches without supervised target tokens before forward",
            "training and evaluation validate supervised target shapes before forward",
            "checkpoint saving rejects invalid step types and scheduler-step mismatches",
            "checkpoint loading rejects scheduler mismatches before mutating model state",
        ),
        gaps=(
            "no decision-grade long-window model quality experiment exists yet",
            "current training evidence is smoke-scale only",
        ),
    ),
    StatusDimension(
        name="backend_performance",
        percent=54,
        evidence=(
            "Delta backend validation compares forward, state and gradients",
            "Delta benchmark matrix reports shape-level throughput and qualification status",
            "MoE benchmark matrix reports routing load and reference throughput",
            "eager Delta and native GQA reduce some reference overhead",
            "global sparse attention uses a sparse causal pattern in the reference path",
            "MoE routed experts use grouped selected-expert matmuls instead of "
            "per-expert token dispatch",
            "MoE router balance is exposed as a differentiable training signal",
            "quantized linear layers store quantized weights with recoverable INT8, "
            "INT4, NF4 and FP8 dequantization contracts",
            "saved quantized artifacts load through schema-checked layer reconstruction",
            "quantized layer application validates targets before mutating model weights",
            "quantization config normalizes modes and rejects unsupported axes",
            "quantized artifact loading rejects incompatible tensor dtypes and metadata shapes",
            "quantized size estimates honor the same explicit skip-layer contract as export",
        ),
        gaps=(
            "Triton/CUDA/Inductor optimized Delta, NSA and MoE kernels are not complete",
            "target-hardware positive speedup is not yet established",
        ),
    ),
    StatusDimension(
        name="zerogpu_and_operations",
        percent=72,
        evidence=(
            "ZeroGPU bundle, adapter limits and persisted evidence are documented",
            "remote quality gate path exists for bounded validation",
            "local GPU preflight reports host GPU, PyTorch CUDA readiness and next actions",
            "local GPU validation plan enumerates CUDA benchmark commands after enablement",
            "local GPU preflight warns when VRAM only supports smoke/operator checks",
            "local GPU operator-memory threshold is configurable in preflight reports",
            "local GPU preflight separates operator and scale-training memory thresholds",
            "local GPU readiness separates CUDA runtime, smoke checks and scale training",
        ),
        gaps=(
            "operations remain tied to Space quota and external Hugging Face state",
            "local CUDA PyTorch runtime still needs enablement on this host",
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
        percent=76,
        evidence=(
            "architecture, training, governance, retrieval and roadmap docs exist",
            "stage-gate and project-hygiene checks are now machine-readable",
            "top-level and subpackage public API exports are protected by snapshot tests",
            "public API reference documents the stable top-level and subpackage exports",
            "inference export rejects unsupported formats instead of silently skipping them",
            "export metadata cannot override model configuration keys",
            "SafeTensors export writes a real safetensors file instead of a torch pickle",
        ),
        gaps=("release packaging and user-facing examples need hardening",),
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
