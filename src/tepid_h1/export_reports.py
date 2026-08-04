"""Benchmark report export utilities for Tepid-H1."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def export_csv(report: dict[str, Any], output_path: str | Path) -> Path:
    """Export a report to CSV format.

    Flattens nested dicts into columns and writes rows.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(_flatten(value, new_key))
            elif isinstance(value, list):
                flat[new_key] = ",".join(str(item) for item in value)
            else:
                flat[new_key] = value
        return flat

    rows = []
    flat = _flatten(report)
    rows.append(flat)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def export_json(report: dict[str, Any], output_path: str | Path) -> Path:
    """Export a report to JSON format with pretty printing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def export_reports(reports: dict[str, dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    """Export multiple reports to both CSV and JSON formats.

    Args:
        reports: Mapping of report name to report data.
        output_dir: Directory to write reports.

    Returns:
        Mapping of report name to output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Path]] = {}
    for name, report in reports.items():
        safe_name = name.replace("/", "_").replace(" ", "_")
        csv_path = output_dir / f"{safe_name}.csv"
        json_path = output_dir / f"{safe_name}.json"
        export_csv(report, csv_path)
        export_json(report, json_path)
        results[name] = {"csv": csv_path, "json": json_path}

    return results
