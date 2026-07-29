from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import spaces

from tepid_h1.integrations import ZeroGPUJobConfig, run_zero_gpu_job

ROOT = Path(__file__).resolve().parent
CORE_REVISION = "336b0d493d4d006fb51e3edf63408ab0d20eb796"


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

    persistent_reports = Path("/data/reports")
    try:
        persistent_reports.mkdir(parents=True, exist_ok=True)
        report_directory = persistent_reports
        storage_kind = "huggingface_bucket_volume"
    except OSError:
        report_directory = Path(tempfile.gettempdir())
        storage_kind = "ephemeral_fallback"
    destination = report_directory / f"tepid-h1-{uuid.uuid4().hex}.json"
    report["deployment_adapter"]["report_storage"] = {
        "kind": storage_kind,
        "path": str(destination),
    }
    destination.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report, str(destination)


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


if __name__ == "__main__":
    demo.queue(max_size=8).launch(allowed_paths=["/data/reports"])
