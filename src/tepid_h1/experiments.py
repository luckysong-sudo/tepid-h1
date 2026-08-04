from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import TepidH1Config
from .data import audit_inventory, load_inventory
from .data.decontamination import file_sha256
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
    trials: int = 1
    batch_size: int = 1
    sequence_length: int = 8
    learning_rate: float = 1e-3
    max_gradient_norm: float = 1.0
    seed: int = 37
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if not 1 <= self.trials <= 20:
            raise ValueError("trials must be between 1 and 20")
        if not 2 <= self.sequence_length <= 64:
            raise ValueError("sequence_length must be between 2 and 64")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16 or float16")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU paired experiments currently require float32")


@dataclass(frozen=True)
class GovernedCorpus:
    batches: tuple[Tensor, ...]
    batch_sha256: str
    start_step: int
    record_ids: tuple[str, ...]
    file_sha256: str
    inventory_file_sha256: str
    inventory_id: str
    source_id: str
    records: int
    domains: tuple[str, ...]


def load_governed_corpus(
    corpus_path: str | Path,
    inventory_path: str | Path,
    config: PairedExperimentConfig,
    *,
    vocab_size: int,
    start_step: int = 0,
) -> GovernedCorpus:
    if not isinstance(start_step, int) or isinstance(start_step, bool) or start_step < 0:
        raise ValueError("start_step must be a non-negative integer")
    corpus_path = Path(corpus_path)
    inventory = load_inventory(inventory_path)
    audit = audit_inventory(inventory)
    if not audit.passed:
        codes = sorted({finding.code for finding in audit.findings if finding.severity == "error"})
        raise ValueError(f"data inventory failed audit: {', '.join(codes)}")

    records: list[dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on corpus line {line_number}") from error
            if not isinstance(item, dict):
                raise TypeError(f"corpus line {line_number} must be an object")
            _validate_corpus_record(
                item,
                line_number=line_number,
                sequence_length=config.sequence_length,
                vocab_size=vocab_size,
            )
            records.append(item)
    if not records:
        raise ValueError("governed corpus must contain at least one record")

    source_ids = {str(record["source_id"]) for record in records}
    if len(source_ids) != 1:
        raise ValueError("governed corpus records must reference exactly one source_id")
    source_id = next(iter(source_ids))
    source = next(
        (
            candidate
            for candidate in inventory.get("sources", [])
            if isinstance(candidate, dict) and candidate.get("id") == source_id
        ),
        None,
    )
    if source is None:
        raise ValueError(f"corpus source_id {source_id!r} is absent from the inventory")

    checksum = file_sha256(corpus_path)
    if source.get("sha256") != checksum:
        raise ValueError(
            f"corpus SHA-256 does not match inventory source {source_id!r}: {checksum}"
        )

    batches = _records_to_batches(records, config, start_step=start_step)
    return GovernedCorpus(
        batches=batches,
        batch_sha256=_batch_digest(batches),
        start_step=start_step,
        record_ids=tuple(str(record["id"]) for record in records),
        file_sha256=checksum,
        inventory_file_sha256=file_sha256(inventory_path),
        inventory_id=audit.inventory_id,
        source_id=source_id,
        records=len(records),
        domains=tuple(sorted({str(record["domain"]) for record in records})),
    )


def validate_governed_split_isolation(
    training: GovernedCorpus,
    validation: GovernedCorpus,
) -> None:
    if training.file_sha256 == validation.file_sha256:
        raise ValueError("training and validation corpus files must be different")
    if training.source_id == validation.source_id:
        raise ValueError("training and validation corpora must use different source_id values")
    overlapping_ids = sorted(set(training.record_ids) & set(validation.record_ids))
    if overlapping_ids:
        raise ValueError(
            "training and validation record IDs overlap: " + ", ".join(overlapping_ids)
        )


def run_paired_smoke(
    config: PairedExperimentConfig,
    *,
    corpus: GovernedCorpus | None = None,
) -> dict[str, Any]:
    model_config = TepidH1Config.smoke()
    baseline_config = TransformerBaselineConfig.active_parameter_matched(model_config)
    batches = (
        corpus.batches
        if corpus is not None
        else _generate_batches(config, model_config.vocab_size)
    )
    device, dtype = _resolve_execution(config)
    execution_batches = tuple(batch.to(device=device) for batch in batches)

    trials = [
        _run_trial(
            config,
            execution_batches,
            model_config=model_config,
            baseline_config=baseline_config,
            trial_index=trial_index,
            device=device,
            dtype=dtype,
        )
        for trial_index in range(config.trials)
    ]
    hybrid_estimate = hybrid_parameter_estimate(model_config)
    baseline_estimate = baseline_parameter_estimate(baseline_config)
    trained_tokens = config.steps * config.batch_size * (config.sequence_length - 1)
    governed = corpus is not None
    data: dict[str, Any] = {
        "kind": "governed_fixed_token_corpus" if governed else "deterministic_random_tokens",
        "batch_sha256": (
            corpus.batch_sha256 if corpus is not None else _batch_digest(batches)
        ),
        "batches": len(batches),
        "tokens_per_model_per_trial": trained_tokens,
        "tokens_per_model_total": trained_tokens * config.trials,
    }
    if corpus is not None:
        data.update(
            {
                "corpus_file_sha256": corpus.file_sha256,
                "inventory_file_sha256": corpus.inventory_file_sha256,
                "inventory_id": corpus.inventory_id,
                "source_id": corpus.source_id,
                "records": corpus.records,
                "domains": list(corpus.domains),
            }
        )

    return {
        "schema_version": 2,
        "experiment": (
            "paired_governed_corpus_smoke" if governed else "paired_random_token_smoke"
        ),
        "interpretation": (
            "Governed fixed-corpus engineering comparison; repeated tiny-CPU results still "
            "do not establish language quality or target-hardware performance."
            if governed
            else "Engineering comparability smoke only; random-token loss and host timing "
            "are not model-quality or production-performance evidence."
        ),
        "environment": _environment_report(device, dtype),
        "config": asdict(config),
        "data": data,
        "trials": trials,
        "aggregates": {
            "method": (
                "arithmetic mean and normal 95% CI; positive ratios use geometric mean "
                "and a log-scale normal 95% CI"
            ),
            "hybrid": _aggregate_model([trial["hybrid"] for trial in trials]),
            "baseline": _aggregate_model([trial["baseline"] for trial in trials]),
            "paired": {
                "baseline_over_hybrid_tokens_per_second": _ratio_statistics(
                    [
                        float(trial["baseline"]["tokens_per_second"])
                        / float(trial["hybrid"]["tokens_per_second"])
                        for trial in trials
                    ]
                ),
                "hybrid_minus_baseline_loss_change": _summary_statistics(
                    [
                        float(trial["hybrid"]["loss_change"])
                        - float(trial["baseline"]["loss_change"])
                        for trial in trials
                    ]
                ),
            },
        },
        "parameters": {
            "hybrid": {
                "actual": trials[0]["hybrid"]["actual_parameters"],
                "estimated_active": hybrid_estimate["active_parameters"],
                "estimated_physical": hybrid_estimate["physical_parameters"],
                "estimate_matches_actual": trials[0]["hybrid"][
                    "parameter_estimate_matches_actual"
                ],
            },
            "baseline": {
                "actual": trials[0]["baseline"]["actual_parameters"],
                "estimated_active": int(baseline_estimate["active_parameters"]),
                "estimated_physical": int(baseline_estimate["physical_parameters"]),
                "estimate_matches_actual": trials[0]["baseline"][
                    "parameter_estimate_matches_actual"
                ],
            },
        },
        "matching": {
            "basis": "per-token active-parameter proxy",
            "baseline_intermediate_size": baseline_config.intermediate_size,
            "active_parameter_gap_percent": baseline_estimate["active_parameter_gap_percent"],
        },
    }


def _validate_corpus_record(
    item: dict[str, Any],
    *,
    line_number: int,
    sequence_length: int,
    vocab_size: int,
) -> None:
    for field in ("id", "source_id", "domain"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            raise ValueError(f"corpus line {line_number} requires non-empty string {field!r}")
    token_ids = item.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) < sequence_length:
        raise ValueError(
            f"corpus line {line_number} requires at least {sequence_length} token_ids"
        )
    if not all(
        isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and 0 <= token_id < vocab_size
        for token_id in token_ids
    ):
        raise ValueError(
            f"corpus line {line_number} token_ids must be integers in [0, {vocab_size})"
        )


def _records_to_batches(
    records: list[dict[str, Any]],
    config: PairedExperimentConfig,
    *,
    start_step: int = 0,
) -> tuple[Tensor, ...]:
    required_records = config.steps * config.batch_size
    start_record = start_step * config.batch_size
    selected = [
        records[(start_record + index) % len(records)] for index in range(required_records)
    ]
    return tuple(
        torch.tensor(
            [
                record["token_ids"][: config.sequence_length]
                for record in selected[
                    step * config.batch_size : (step + 1) * config.batch_size
                ]
            ],
            dtype=torch.long,
        )
        for step in range(config.steps)
    )


def _run_trial(
    config: PairedExperimentConfig,
    batches: tuple[Tensor, ...],
    *,
    model_config: TepidH1Config,
    baseline_config: TransformerBaselineConfig,
    trial_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    trial_seed = config.seed + trial_index
    torch.manual_seed(trial_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(trial_seed)
    hybrid = TepidH1CausalLM(model_config).to(device=device, dtype=dtype)
    torch.manual_seed(trial_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(trial_seed)
    baseline = TransformerBaselineCausalLM(baseline_config).to(device=device, dtype=dtype)
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
        order_offset = step + trial_index
        order = ["hybrid", "baseline"] if order_offset % 2 == 0 else ["baseline", "hybrid"]
        execution_order.append(order)
        for name in order:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            metrics = causal_lm_train_step(
                models[name],
                input_ids,
                optimizers[name],
                max_gradient_norm=config.max_gradient_norm,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak_memory = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            )
            measurements[name].append(
                {
                    **asdict(metrics),
                    "elapsed_seconds": elapsed,
                    "peak_memory_bytes": peak_memory,  # type: ignore[dict-item]
                }
            )

    hybrid_estimate = hybrid_parameter_estimate(model_config)
    baseline_estimate = baseline_parameter_estimate(baseline_config)
    return {
        "index": trial_index,
        "seed": trial_seed,
        "execution_order": execution_order,
        "hybrid": _summarize(
            measurements["hybrid"],
            actual_parameters=_parameter_count(hybrid),
            estimated_physical_parameters=hybrid_estimate["physical_parameters"],
        ),
        "baseline": _summarize(
            measurements["baseline"],
            actual_parameters=_parameter_count(baseline),
            estimated_physical_parameters=int(baseline_estimate["physical_parameters"]),
        ),
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
    measurements: list[dict[str, Any]],
    *,
    actual_parameters: int,
    estimated_physical_parameters: int,
) -> dict[str, Any]:
    elapsed = sum(float(item["elapsed_seconds"]) for item in measurements)
    trained_tokens = sum(int(item["trained_tokens"]) for item in measurements)
    losses = [float(item["loss"]) for item in measurements]
    peak_memory_values = [
        int(item["peak_memory_bytes"])
        for item in measurements
        if item["peak_memory_bytes"] is not None
    ]
    return {
        "actual_parameters": actual_parameters,
        "parameter_estimate_matches_actual": actual_parameters == estimated_physical_parameters,
        "trained_tokens": trained_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": trained_tokens / elapsed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_change": losses[-1] - losses[0],
        "peak_memory_bytes": max(peak_memory_values) if peak_memory_values else None,
        "steps": measurements,
    }


def _aggregate_model(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "trials": len(summaries),
        "trained_tokens_per_trial": int(summaries[0]["trained_tokens"]),
        "tokens_per_second": _summary_statistics(
            [float(summary["tokens_per_second"]) for summary in summaries]
        ),
        "initial_loss": _summary_statistics(
            [float(summary["initial_loss"]) for summary in summaries]
        ),
        "final_loss": _summary_statistics(
            [float(summary["final_loss"]) for summary in summaries]
        ),
        "loss_change": _summary_statistics(
            [float(summary["loss_change"]) for summary in summaries]
        ),
    }
    peak_memory_values = [
        float(summary["peak_memory_bytes"])
        for summary in summaries
        if summary["peak_memory_bytes"] is not None
    ]
    result["peak_memory_bytes"] = (
        _summary_statistics(peak_memory_values) if peak_memory_values else None
    )
    return result


def _summary_statistics(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * standard_deviation / math.sqrt(len(values))
    return {
        "samples": len(values),
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def _ratio_statistics(values: list[float]) -> dict[str, float | int]:
    if any(value <= 0 for value in values):
        raise ValueError("ratio statistics require positive values")
    log_values = [math.log(value) for value in values]
    log_mean = statistics.fmean(log_values)
    log_standard_deviation = statistics.stdev(log_values) if len(values) > 1 else 0.0
    log_margin = 1.96 * log_standard_deviation / math.sqrt(len(values))
    return {
        "samples": len(values),
        "geometric_mean": math.exp(log_mean),
        "log_sample_standard_deviation": log_standard_deviation,
        "ci95_low": math.exp(log_mean - log_margin),
        "ci95_high": math.exp(log_mean + log_margin),
    }


def _resolve_execution(
    config: PairedExperimentConfig,
) -> tuple[torch.device, torch.dtype]:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if (
        config.device == "cuda"
        and config.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 was requested but the CUDA device does not support it")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]
    return torch.device(config.device), dtype


def _environment_report(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device_type": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "timing_scope": "synchronized train step; data batches are preloaded on device",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update(
            {
                "device_name": properties.name,
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": properties.total_memory,
                "cuda_runtime": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return report
