"""Tests for report export utilities."""
import json
from pathlib import Path

import pytest

from tepid_h1.export_reports import export_csv, export_json, export_reports


class TestExportReports:
    def test_export_csv_flattens_nested_dict(self, tmp_path: Path) -> None:
        report = {"schema_version": 1, "timing": {"ref": 1.5, "cand": 2.0}}
        csv_path = tmp_path / "report.csv"
        result = export_csv(report, csv_path)

        assert result.exists()
        content = result.read_text()
        assert "schema_version" in content
        assert "timing_ref" in content
        assert "1.5" in content

    def test_export_json_pretty_prints(self, tmp_path: Path) -> None:
        report = {"schema_version": 1, "result": "passed"}
        json_path = tmp_path / "report.json"
        result = export_json(report, json_path)

        assert result.exists()
        data = json.loads(result.read_text())
        assert data["schema_version"] == 1
        assert data["result"] == "passed"

    def test_export_reports_multiple_formats(self, tmp_path: Path) -> None:
        reports = {
            "moe_balance": {"load_cv": 0.04, "passed": True},
            "delta": {"numerical_passed": True, "speedup": 1.33},
        }
        results = export_reports(reports, tmp_path)

        assert "moe_balance" in results
        assert "delta" in results
        assert results["moe_balance"]["csv"].exists()
        assert results["moe_balance"]["json"].exists()
        assert results["delta"]["csv"].exists()
        assert results["delta"]["json"].exists()

    def test_export_csv_handles_list_values(self, tmp_path: Path) -> None:
        report = {"schema_version": 1, "sources": ["zh", "en", "code"]}
        csv_path = tmp_path / "report.csv"
        result = export_csv(report, csv_path)

        assert result.exists()
        content = result.read_text()
        assert "zh,en,code" in content

    def test_export_csv_creates_parent_dirs(self, tmp_path: Path) -> None:
        report = {"schema_version": 1}
        nested_path = tmp_path / "nested" / "dir" / "report.csv"
        result = export_csv(report, nested_path)

        assert result.exists()
        assert result.parent == nested_path.parent
