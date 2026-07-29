from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from gradio_client import Client
from huggingface_hub import HfApi

SPACE_ID = "himartoffice/Tepid-H1"
SPACE_BUNDLE = Path("integrations/huggingface-zero-gpu")
DASHBOARD_COMMIT_API = (
    "https://api.github.com/repos/luckysong-sudo/tepid-h1-progress/commits/main"
)
DEV_MODE_API = f"https://huggingface.co/api/spaces/{SPACE_ID}/dev-mode"
SPACE_INFO_API = f"https://huggingface.co/api/spaces/{SPACE_ID}"
REMOTE_GATE_API = "/run_remote_quality_gate"


def required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable {name} is missing")
    return value


def get_dashboard_revision(session: requests.Session) -> str:
    response = session.get(DASHBOARD_COMMIT_API, timeout=30)
    response.raise_for_status()
    revision = response.json().get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("GitHub returned an invalid dashboard revision")
    return revision


def prepare_bundle(destination: Path, core_revision: str, dashboard_revision: str) -> None:
    shutil.copytree(SPACE_BUNDLE, destination)
    app_path = destination / "app.py"
    requirements_path = destination / "requirements.txt"
    app_source = app_path.read_text(encoding="utf-8")
    app_source, core_replacements = re.subn(
        r'CORE_REVISION = "[0-9a-f]{40}"',
        f'CORE_REVISION = "{core_revision}"',
        app_source,
        count=1,
    )
    app_source, dashboard_replacements = re.subn(
        r'DASHBOARD_REVISION = "[0-9a-f]{40}"',
        f'DASHBOARD_REVISION = "{dashboard_revision}"',
        app_source,
        count=1,
    )
    requirements = requirements_path.read_text(encoding="utf-8")
    requirements, requirement_replacements = re.subn(
        r"(tepid-h1 @ git\+https://github\.com/luckysong-sudo/tepid-h1\.git@)"
        r"[0-9a-f]{40}",
        rf"\g<1>{core_revision}",
        requirements,
        count=1,
    )
    if (core_replacements, dashboard_replacements, requirement_replacements) != (1, 1, 1):
        raise RuntimeError("Space bundle revision markers are missing or ambiguous")
    app_path.write_text(app_source, encoding="utf-8")
    requirements_path.write_text(requirements, encoding="utf-8")


def space_info(session: requests.Session) -> dict[str, Any]:
    response = session.get(SPACE_INFO_API, timeout=30)
    response.raise_for_status()
    return response.json()


def set_dev_mode(session: requests.Session, enabled: bool) -> None:
    response = session.post(DEV_MODE_API, json={"enabled": enabled}, timeout=60)
    response.raise_for_status()


def wait_for_runtime(
    session: requests.Session,
    *,
    source_revision: str,
    dev_mode: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        info = space_info(session)
        runtime = info.get("runtime") or {}
        last_state = {
            "source_revision": info.get("sha"),
            "runtime_revision": runtime.get("sha"),
            "stage": runtime.get("stage"),
            "dev_mode": runtime.get("devMode"),
        }
        if (
            info.get("sha") == source_revision
            and runtime.get("sha") == source_revision
            and runtime.get("stage") == "RUNNING"
            and runtime.get("devMode") is dev_mode
        ):
            return last_state
        time.sleep(10)
    raise TimeoutError(f"Space runtime did not converge: {last_state}")


def refresh_dev_mode(session: requests.Session, source_revision: str) -> dict[str, Any]:
    disabled = False
    try:
        set_dev_mode(session, False)
        disabled = True
        wait_for_runtime(
            session,
            source_revision=source_revision,
            dev_mode=False,
            timeout_seconds=900,
        )
        set_dev_mode(session, True)
        disabled = False
        return wait_for_runtime(
            session,
            source_revision=source_revision,
            dev_mode=True,
            timeout_seconds=600,
        )
    finally:
        if disabled:
            try:
                set_dev_mode(session, True)
            except requests.RequestException:
                pass


def wait_for_quality_api(hf_token: str, timeout_seconds: int = 300) -> Client:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client = Client(SPACE_ID, token=hf_token, verbose=False)
            endpoints = client.view_api(return_format="dict").get("named_endpoints", {})
            if REMOTE_GATE_API in endpoints:
                return client
        except Exception as error:
            last_error = error
        time.sleep(10)
    raise TimeoutError(f"remote quality API did not become ready: {last_error}")


def run_quality_gate(client: Client, report_path: Path) -> dict[str, Any]:
    result = client.submit(api_name=REMOTE_GATE_API).result(timeout=720)
    if not isinstance(result, (tuple, list)) or not result:
        raise RuntimeError("remote quality gate returned an invalid response")
    report = result[0]
    if isinstance(report, str):
        report = json.loads(report)
    if not isinstance(report, dict):
        raise RuntimeError("remote quality report is not a JSON object")
    report_path.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    return report


def append_summary(
    *,
    core_revision: str,
    dashboard_revision: str,
    space_revision: str,
    runtime: dict[str, Any],
    report: dict[str, Any],
) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    failed = [check["name"] for check in report.get("checks", []) if not check.get("passed")]
    lines = [
        "## Tepid-H1 ZeroGPU deployment",
        "",
        f"- Core revision: `{core_revision}`",
        f"- Dashboard revision: `{dashboard_revision}`",
        f"- Space revision: `{space_revision}`",
        f"- Runtime revision: `{runtime.get('runtime_revision')}`",
        f"- Dev Mode restored: `{runtime.get('dev_mode')}`",
        f"- Remote quality gate: `{'passed' if report.get('passed') else 'failed'}`",
    ]
    if failed:
        lines.append(f"- Failed checks: `{', '.join(failed)}`")
    Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    hf_token = required_environment("HF_TOKEN")
    github_token = required_environment("GH_TOKEN")
    core_revision = required_environment("CORE_REVISION")
    report_path = Path(required_environment("QUALITY_REPORT_PATH"))
    if not re.fullmatch(r"[0-9a-f]{40}", core_revision):
        raise RuntimeError("CORE_REVISION must be a full Git commit SHA")

    github_session = requests.Session()
    github_session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    dashboard_revision = get_dashboard_revision(github_session)

    with tempfile.TemporaryDirectory(prefix="tepid-h1-space-") as temporary_directory:
        bundle = Path(temporary_directory) / "bundle"
        prepare_bundle(bundle, core_revision, dashboard_revision)
        commit = HfApi(token=hf_token).upload_folder(
            repo_id=SPACE_ID,
            repo_type="space",
            folder_path=bundle,
            commit_message=f"Deploy core {core_revision[:7]} and dashboard {dashboard_revision[:7]}",
        )
    space_revision = commit.oid
    if not isinstance(space_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", space_revision):
        raise RuntimeError("Hugging Face returned an invalid Space revision")

    hf_session = requests.Session()
    hf_session.headers.update({"Authorization": f"Bearer {hf_token}"})
    runtime = refresh_dev_mode(hf_session, space_revision)
    client = wait_for_quality_api(hf_token)
    report = run_quality_gate(client, report_path)
    append_summary(
        core_revision=core_revision,
        dashboard_revision=dashboard_revision,
        space_revision=space_revision,
        runtime=runtime,
        report=report,
    )
    if report.get("core_revision") != core_revision:
        raise RuntimeError("remote report core revision does not match deployed revision")
    if report.get("dashboard_revision") != dashboard_revision:
        raise RuntimeError("remote report dashboard revision does not match deployed revision")
    if not report.get("passed"):
        failed = [check["name"] for check in report.get("checks", []) if not check.get("passed")]
        raise RuntimeError(f"remote ZeroGPU quality gate failed: {failed}")


if __name__ == "__main__":
    main()
