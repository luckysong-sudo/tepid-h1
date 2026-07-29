from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import spaces
import torch

from tepid_h1.integrations import ZeroGPUJobConfig, run_zero_gpu_job

ROOT = Path(__file__).resolve().parent
CORE_REVISION = "a383613c9d71d10fb0d4407a5d1e0dc1ec462034"
CORE_REPOSITORY = "https://github.com/luckysong-sudo/tepid-h1.git"
DASHBOARD_REVISION = "c62c5a17543451d7bd81ce302c74a14949da4e2b"
DASHBOARD_REPOSITORY = "https://github.com/luckysong-sudo/tepid-h1-progress.git"
QUALITY_GATE_TIMEOUT_SECONDS = 290


def _persist_report(report: dict, prefix: str) -> str:
    persistent_reports = Path("/data/reports")
    try:
        persistent_reports.mkdir(parents=True, exist_ok=True)
        report_directory = persistent_reports
        storage_kind = "huggingface_bucket_volume"
    except OSError:
        report_directory = Path(tempfile.gettempdir())
        storage_kind = "ephemeral_fallback"
    destination = report_directory / f"{prefix}-{uuid.uuid4().hex}.json"
    report["deployment_adapter"]["report_storage"] = {
        "kind": storage_kind,
        "path": str(destination),
    }
    destination.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return str(destination)


@spaces.GPU(duration=60)
def run_gpu_experiment(steps: int, trials: int, dtype: str):
    try:
        if isinstance(steps, bool) or int(steps) != steps:
            raise TypeError("steps must be an integer")
        if isinstance(trials, bool) or int(trials) != trials:
            raise TypeError("trials must be an integer")
        report = run_zero_gpu_job(
            ZeroGPUJobConfig(steps=int(steps), trials=int(trials), dtype=dtype),
            corpus_path=ROOT / "paired_corpus.jsonl",
            inventory_path=ROOT / "data_inventory.json",
            device="cuda",
            core_revision=CORE_REVISION,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    destination = _persist_report(report, "tepid-h1")
    return report, destination


def _run_check(
    name: str,
    command: list[str],
    *,
    working_directory: Path,
    deadline: float,
) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {
            "name": name,
            "passed": False,
            "returncode": None,
            "elapsed_seconds": 0.0,
            "output": "quality-gate deadline exhausted before this check",
        }
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=working_directory,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=min(120.0, remaining),
            check=False,
        )
        output = completed.stdout[-4_000:]
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        captured = error.stdout or ""
        if isinstance(captured, bytes):
            captured = captured.decode(errors="replace")
        output = f"{captured[-3_800:]}\ncheck timed out"
        returncode = None
    except OSError as error:
        output = f"{type(error).__name__}: {error}"
        returncode = None
    return {
        "name": name,
        "passed": returncode == 0,
        "returncode": returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "output": output,
    }


@spaces.GPU(duration=300)
def run_remote_quality_gate():
    started = time.perf_counter()
    deadline = time.monotonic() + QUALITY_GATE_TIMEOUT_SECONDS
    checks: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="tepid-h1-quality-") as temporary_directory:
        root = Path(temporary_directory)
        checkout = root / "source"
        virtual_environment = root / "venv"
        checks.append(
            _run_check(
                "fetch_pinned_source",
                [
                    "git",
                    "clone",
                    CORE_REPOSITORY,
                    str(checkout),
                ],
                working_directory=root,
                deadline=deadline,
            )
        )
        if checks[-1]["passed"]:
            checks.append(
                _run_check(
                    "checkout_pinned_revision",
                    [
                        "git",
                        "checkout",
                        "--detach",
                        CORE_REVISION,
                    ],
                    working_directory=checkout,
                    deadline=deadline,
                )
            )
        if checks[-1]["passed"]:
            checks.append(
                _run_check(
                    "create_isolated_environment",
                    [
                        sys.executable,
                        "-m",
                        "venv",
                        "--system-site-packages",
                        str(virtual_environment),
                    ],
                    working_directory=checkout,
                    deadline=deadline,
                )
            )
        environment_python = virtual_environment / "bin" / "python"
        environment_cli = virtual_environment / "bin" / "tepid-h1"
        if checks[-1]["passed"]:
            checks.append(
                _run_check(
                    "install_pinned_source",
                    [
                        str(environment_python),
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--editable",
                        ".",
                    ],
                    working_directory=checkout,
                    deadline=deadline,
                )
            )

        if checks[-1]["passed"]:
            commands = [
                (
                    "ruff",
                    [
                        str(environment_python),
                        "-m",
                        "ruff",
                        "check",
                        "src",
                        "tests",
                        "integrations/huggingface-zero-gpu/app.py",
                    ],
                ),
                (
                    "unit_tests",
                    [
                        str(environment_python),
                        "-m",
                        "unittest",
                        "discover",
                        "-s",
                        "tests",
                        "-p",
                        "test_*.py",
                        "-v",
                    ],
                ),
                (
                    "data_audit",
                    [
                        str(environment_cli),
                        "data-audit",
                        "configs/data_inventory.example.json",
                    ],
                ),
                (
                    "decontamination",
                    [
                        str(environment_cli),
                        "decontaminate",
                        "--training",
                        "configs/decontamination_training.example.jsonl",
                        "--benchmark",
                        "configs/decontamination_benchmark.example.jsonl",
                    ],
                ),
                (
                    "governed_train_smoke",
                    [
                        str(environment_cli),
                        "train-smoke",
                        "--steps",
                        "1",
                        "--corpus",
                        "configs/paired_corpus.example.jsonl",
                        "--validation-corpus",
                        "configs/validation_corpus.example.jsonl",
                        "--validation-steps",
                        "3",
                        "--inventory",
                        "configs/data_inventory.example.json",
                        "--checkpoint",
                        str(root / "tepid-h1-smoke.pt"),
                        "--report",
                        str(root / "governed-train.json"),
                    ],
                ),
                (
                    "governed_train_resume",
                    [
                        str(environment_cli),
                        "train-smoke",
                        "--steps",
                        "1",
                        "--corpus",
                        "configs/paired_corpus.example.jsonl",
                        "--validation-corpus",
                        "configs/validation_corpus.example.jsonl",
                        "--validation-steps",
                        "3",
                        "--inventory",
                        "configs/data_inventory.example.json",
                        "--checkpoint",
                        str(root / "tepid-h1-smoke.pt"),
                        "--resume",
                        "--report",
                        str(root / "governed-resume.json"),
                    ],
                ),
                (
                    "retrieval_generate",
                    [
                        str(environment_cli),
                        "retrieval-generate",
                        "--prompts",
                        str(root / "retrieval-prompts.jsonl"),
                        "--answers",
                        str(root / "retrieval-answers.jsonl"),
                    ],
                ),
                (
                    "retrieval_score",
                    [
                        str(environment_cli),
                        "retrieval-score",
                        "--answers",
                        str(root / "retrieval-answers.jsonl"),
                        "--predictions",
                        str(root / "retrieval-answers.jsonl"),
                    ],
                ),
                (
                    "baseline_report",
                    [str(environment_cli), "baseline-report", "--variant", "reference"],
                ),
                (
                    "delta_validation",
                    [
                        str(environment_cli),
                        "delta-validate",
                        "--backend",
                        "eager",
                        "--device",
                        "cpu",
                        "--dtype",
                        "float32",
                        "--iterations",
                        "1",
                        "--report",
                        str(root / "delta-compiler-boundary.json"),
                    ],
                ),
                (
                    "paired_cuda_smoke",
                    [
                        str(environment_cli),
                        "compare-smoke",
                        "--steps",
                        "1",
                        "--trials",
                        "1",
                        "--device",
                        "cuda",
                        "--dtype",
                        "bfloat16",
                        "--corpus",
                        "configs/paired_corpus.example.jsonl",
                        "--inventory",
                        "configs/data_inventory.example.json",
                        "--report",
                        str(root / "paired-governed.json"),
                    ],
                ),
            ]
            for name, command in commands:
                checks.append(
                    _run_check(
                        name,
                        command,
                        working_directory=checkout,
                        deadline=deadline,
                    )
                )

        dashboard_checkout = root / "dashboard"
        checks.append(
            _run_check(
                "fetch_dashboard_source",
                [
                    "git",
                    "clone",
                    DASHBOARD_REPOSITORY,
                    str(dashboard_checkout),
                ],
                working_directory=root,
                deadline=deadline,
            )
        )
        if checks[-1]["passed"]:
            checks.append(
                _run_check(
                    "checkout_dashboard_revision",
                    [
                        "git",
                        "checkout",
                        "--detach",
                        DASHBOARD_REVISION,
                    ],
                    working_directory=dashboard_checkout,
                    deadline=deadline,
                )
            )
        if checks[-1]["passed"]:
            checks.append(
                _run_check(
                    "dashboard_dependencies",
                    ["npm", "ci", "--ignore-scripts"],
                    working_directory=dashboard_checkout,
                    deadline=deadline,
                )
            )
        if checks[-1]["passed"]:
            checks.append(
                _run_check(
                    "dashboard_checks",
                    ["npm", "run", "check"],
                    working_directory=dashboard_checkout,
                    deadline=deadline,
                )
            )

    report = {
        "schema_version": 1,
        "experiment": "remote_zero_gpu_quality_gate",
        "core_revision": CORE_REVISION,
        "dashboard_revision": DASHBOARD_REVISION,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "cuda_runtime": torch.version.cuda,
        },
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "deployment_adapter": {
            "name": "huggingface_zerogpu_gradio",
            "schema_version": 1,
            "report_storage": None,
        },
    }
    destination = _persist_report(report, "tepid-h1-quality")
    return report, destination


with gr.Blocks(title="Tepid-H1 ZeroGPU Qualification") as demo:
    gr.Markdown(
        """
        # Tepid-H1 · ZeroGPU qualification

        Run the governed tiny hybrid/baseline comparison inside a real CUDA allocation.
        This is a compatibility and measurement smoke test, not a model-quality claim.
        """
    )
    with gr.Row():
        steps = gr.Slider(1, 5, value=2, step=1, label="Training steps")
        trials = gr.Slider(1, 3, value=2, step=1, label="Repeated trials")
        dtype = gr.Dropdown(
            choices=("bfloat16", "float16", "float32"),
            value="bfloat16",
            label="Parameter dtype",
        )
    run_button = gr.Button("Run on ZeroGPU", variant="primary")
    report_view = gr.JSON(label="Qualification report")
    report_file = gr.File(label="Download JSON report")
    run_button.click(
        fn=run_gpu_experiment,
        inputs=(steps, trials, dtype),
        outputs=(report_view, report_file),
        api_name="run_gpu_experiment",
    )
    gr.Markdown(
        """
        ## Remote full quality gate

        Run lint, all unit tests, governance checks, training/checkpoint checks and a
        governed CUDA smoke entirely inside the ZeroGPU allocation. The host workspace
        does not execute these tests.
        """
    )
    quality_button = gr.Button("Run full remote quality gate")
    quality_report_view = gr.JSON(label="Remote quality report")
    quality_report_file = gr.File(label="Download remote quality JSON")
    quality_button.click(
        fn=run_remote_quality_gate,
        inputs=(),
        outputs=(quality_report_view, quality_report_file),
        api_name="run_remote_quality_gate",
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(allowed_paths=["/data/reports"])
