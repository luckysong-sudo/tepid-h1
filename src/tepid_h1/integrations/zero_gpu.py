from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import TepidH1Config
from ..experiments import (
    PairedExperimentConfig,
    load_governed_corpus,
    run_paired_smoke,
)


@dataclass(frozen=True)
class ZeroGPUJobConfig:
    steps: int = 2
    trials: int = 2
    dtype: str = "bfloat16"
    seed: int = 37

    def __post_init__(self) -> None:
        if not isinstance(self.steps, int) or isinstance(self.steps, bool):
            raise TypeError("ZeroGPU steps must be an integer")
        if not isinstance(self.trials, int) or isinstance(self.trials, bool):
            raise TypeError("ZeroGPU trials must be an integer")
        if not 1 <= self.steps <= 5:
            raise ValueError("ZeroGPU steps must be between 1 and 5")
        if not 1 <= self.trials <= 3:
            raise ValueError("ZeroGPU trials must be between 1 and 3")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("ZeroGPU dtype must be float32, bfloat16 or float16")


def run_zero_gpu_job(
    job: ZeroGPUJobConfig,
    *,
    corpus_path: str | Path,
    inventory_path: str | Path,
    device: str = "cuda",
    core_revision: str | None = None,
) -> dict[str, Any]:
    experiment_config = PairedExperimentConfig(
        steps=job.steps,
        trials=job.trials,
        batch_size=1,
        sequence_length=8,
        learning_rate=1e-3,
        max_gradient_norm=1.0,
        seed=job.seed,
        device=device,
        dtype=job.dtype,
    )
    corpus = load_governed_corpus(
        corpus_path,
        inventory_path,
        experiment_config,
        vocab_size=TepidH1Config.smoke().vocab_size,
    )
    report = run_paired_smoke(experiment_config, corpus=corpus)
    report.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "deployment_adapter": {
                "name": "huggingface_zerogpu_gradio",
                "schema_version": 1,
                "core_revision": core_revision,
                "limits": {
                    "maximum_steps": 5,
                    "maximum_trials": 3,
                    "fixed_batch_size": 1,
                    "fixed_sequence_length": 8,
                },
            },
        }
    )
    return report
