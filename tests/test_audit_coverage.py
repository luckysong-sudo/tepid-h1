"""Tests for data audit module coverage."""
import json
from pathlib import Path

import pytest

from tepid_h1.data.audit import (
    AuditFinding,
    AuditReport,
    audit_inventory,
    load_inventory,
)


class TestAuditReport:
    def test_report_to_dict(self):
        report = AuditReport(
            inventory_id="test-inv",
            passed=True,
            source_count=2,
            approved_source_count=2,
            estimated_tokens=1000,
            findings=(),
        )
        d = report.to_dict()
        assert d["inventory_id"] == "test-inv"
        assert d["passed"] is True
        assert d["source_count"] == 2


class TestAuditInventory:
    def test_audit_valid_inventory(self, tmp_path: Path) -> None:
        inventory = {
            "schema_version": 1,
            "inventory_id": "test",
            "owner": "test-owner",
            "generated_at": "2026-01-01T00:00:00Z",
            "sources": [
                {
                    "id": "src1",
                    "name": "Source 1",
                    "uri": "repo://src1",
                    "snapshot": "v1",
                    "sha256": "a" * 64,
                    "license_id": "MIT",
                    "license_category": "permissive",
                    "commercial_use": True,
                    "rights_evidence": "contract",
                    "languages": ["zh"],
                    "domains": ["general"],
                    "estimated_tokens": 1000,
                    "pii_status": "absent",
                    "quality_status": "accepted",
                }
            ],
            "repository_decontamination": {
                "status": "complete",
                "method": "hash comparison",
                "benchmark_sets": ["heldout"],
                "report_uri": "docs/decontamination.md",
                "completed_at": "2026-01-01T00:00:00Z",
            },
        }
        path = tmp_path / "inventory.json"
        path.write_text(json.dumps(inventory))
        report = audit_inventory(load_inventory(path))
        assert report.passed
        assert len(report.findings) == 0

    def test_audit_missing_required_fields(self, tmp_path: Path) -> None:
        inventory = {"schema_version": 1}
        path = tmp_path / "inventory.json"
        path.write_text(json.dumps(inventory))
        report = audit_inventory(load_inventory(path))
        assert not report.passed
        assert len(report.findings) > 0

    def test_audit_invalid_license(self, tmp_path: Path) -> None:
        inventory = {
            "schema_version": 1,
            "inventory_id": "test",
            "owner": "owner",
            "generated_at": "2026-01-01T00:00:00Z",
            "sources": [
                {
                    "id": "src1",
                    "name": "Source 1",
                    "uri": "repo://src1",
                    "snapshot": "v1",
                    "sha256": "a" * 64,
                    "license_id": "GPL-3.0",
                    "license_category": "restricted",
                    "commercial_use": True,
                    "rights_evidence": "contract",
                    "languages": ["zh"],
                    "domains": ["general"],
                    "estimated_tokens": 1000,
                    "pii_status": "absent",
                    "quality_status": "accepted",
                }
            ],
            "repository_decontamination": {
                "status": "complete",
                "method": "hash comparison",
                "benchmark_sets": ["heldout"],
                "report_uri": "docs/decontamination.md",
                "completed_at": "2026-01-01T00:00:00Z",
            },
        }
        path = tmp_path / "inventory.json"
        path.write_text(json.dumps(inventory))
        report = audit_inventory(load_inventory(path))
        assert not report.passed

    def test_audit_pii_present(self, tmp_path: Path) -> None:
        inventory = {
            "schema_version": 1,
            "inventory_id": "test",
            "owner": "owner",
            "generated_at": "2026-01-01T00:00:00Z",
            "sources": [
                {
                    "id": "src1",
                    "name": "Source 1",
                    "uri": "repo://src1",
                    "snapshot": "v1",
                    "sha256": "a" * 64,
                    "license_id": "MIT",
                    "license_category": "permissive",
                    "commercial_use": True,
                    "rights_evidence": "contract",
                    "languages": ["zh"],
                    "domains": ["general"],
                    "estimated_tokens": 1000,
                    "pii_status": "present",
                    "quality_status": "accepted",
                }
            ],
            "repository_decontamination": {
                "status": "complete",
                "method": "hash comparison",
                "benchmark_sets": ["heldout"],
                "report_uri": "docs/decontamination.md",
                "completed_at": "2026-01-01T00:00:00Z",
            },
        }
        path = tmp_path / "inventory.json"
        path.write_text(json.dumps(inventory))
        report = audit_inventory(load_inventory(path))
        assert not report.passed

    def test_load_inventory_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "inventory.json"
        path.write_text("not json")
        with pytest.raises(Exception):
            load_inventory(path)

    def test_load_inventory_root_not_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "inventory.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(TypeError):
            load_inventory(path)
