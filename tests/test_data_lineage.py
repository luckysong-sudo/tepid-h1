"""Tests for data lineage tracking and license compatibility checking."""

import pytest

from tepid_h1.data import (
    LicenseCompatibilityReport,
    LineageEntry,
    LineageReport,
    LineageTracker,
    check_license_compatibility,
)


class TestLineageTracker:
    """Test the LineageTracker class."""

    def test_register_root_source(self):
        tracker = LineageTracker()
        tracker.register_root_source("raw_data_v1")
        assert "raw_data_v1" in tracker.registered_sources
        assert "raw_data_v1" in tracker.root_sources

    def test_rejects_duplicate_root_source(self):
        tracker = LineageTracker()
        tracker.register_root_source("raw_v1")
        with pytest.raises(ValueError, match="already registered"):
            tracker.register_root_source("raw_v1")

    def test_rejects_empty_source_id(self):
        tracker = LineageTracker()
        with pytest.raises(ValueError, match="not be empty"):
            tracker.register_root_source("")

    def test_record_transformation(self):
        tracker = LineageTracker()
        tracker.register_root_source("raw_a")
        tracker.register_root_source("raw_b")
        entry = tracker.record_transformation(
            stage="dedup",
            input_source_ids=("raw_a", "raw_b"),
            output_source_id="deduped_v1",
            transformation="deduplicate",
            operator="pipeline",
        )
        assert isinstance(entry, LineageEntry)
        assert entry.stage == "dedup"
        assert entry.input_source_ids == ("raw_a", "raw_b")
        assert entry.output_source_id == "deduped_v1"
        assert entry.transformation == "deduplicate"
        assert entry.operator == "pipeline"
        assert "deduped_v1" in tracker.registered_sources

    def test_rejects_unregistered_input(self):
        tracker = LineageTracker()
        with pytest.raises(ValueError, match="not registered"):
            tracker.record_transformation(
                stage="clean",
                input_source_ids=("unknown_source",),
                output_source_id="clean_v1",
                transformation="clean",
            )

    def test_rejects_duplicate_output(self):
        tracker = LineageTracker()
        tracker.register_root_source("raw_a")
        tracker.record_transformation(
            stage="clean",
            input_source_ids=("raw_a",),
            output_source_id="clean_v1",
            transformation="clean",
        )
        with pytest.raises(ValueError, match="already registered"):
            tracker.record_transformation(
                stage="dedup",
                input_source_ids=("raw_a",),
                output_source_id="clean_v1",
                transformation="dedup",
            )

    def test_rejects_empty_inputs(self):
        tracker = LineageTracker()
        with pytest.raises(ValueError, match="not be empty"):
            tracker.record_transformation(
                stage="clean",
                input_source_ids=(),
                output_source_id="clean_v1",
                transformation="clean",
            )

    def test_get_lineage_for_root(self):
        tracker = LineageTracker()
        tracker.register_root_source("raw_v1")
        report = tracker.get_lineage("raw_v1")
        assert isinstance(report, LineageReport)
        assert report.source_id == "raw_v1"
        assert report.chain_length == 0
        assert report.root_source_ids == ("raw_v1",)
        assert report.passed

    def test_get_lineage_for_transformed_source(self):
        tracker = LineageTracker()
        tracker.register_root_source("raw_a")
        tracker.register_root_source("raw_b")
        tracker.record_transformation(
            stage="merge",
            input_source_ids=("raw_a", "raw_b"),
            output_source_id="merged_v1",
            transformation="merge_and_dedup",
        )
        report = tracker.get_lineage("merged_v1")
        assert report.source_id == "merged_v1"
        assert report.chain_length == 1
        assert set(report.root_source_ids) == {"raw_a", "raw_b"}
        assert report.passed

    def test_get_lineage_multi_hop(self):
        tracker = LineageTracker()
        tracker.register_root_source("raw_a")
        tracker.record_transformation(
            stage="clean",
            input_source_ids=("raw_a",),
            output_source_id="clean_v1",
            transformation="clean",
        )
        tracker.record_transformation(
            stage="tokenize",
            input_source_ids=("clean_v1",),
            output_source_id="tokenized_v1",
            transformation="tokenize",
        )
        report = tracker.get_lineage("tokenized_v1")
        assert report.chain_length == 2
        assert report.root_source_ids == ("raw_a",)
        assert report.passed

    def test_get_lineage_unregistered_source(self):
        tracker = LineageTracker()
        with pytest.raises(ValueError, match="not registered"):
            tracker.get_lineage("unknown")

    def test_lineage_report_to_dict(self):
        tracker = LineageTracker()
        tracker.register_root_source("raw_v1")
        report = tracker.get_lineage("raw_v1")
        payload = report.to_dict()
        assert payload["schema_version"] == 1
        assert payload["source_id"] == "raw_v1"
        assert payload["chain_length"] == 0
        assert payload["passed"] is True


class TestLicenseCompatibility:
    """Test the check_license_compatibility function."""

    def test_all_compatible_sources_pass(self):
        sources = [
            {"id": "s1", "license_category": "permissive"},
            {"id": "s2", "license_category": "public_domain"},
            {"id": "s3", "license_category": "contracted"},
        ]
        report = check_license_compatibility(sources, usage_context="commercial")
        assert isinstance(report, LicenseCompatibilityReport)
        assert report.passed
        assert report.compatible_sources == 3
        assert report.incompatible_sources == 0

    def test_restricted_license_fails_commercial(self):
        sources = [
            {"id": "s1", "license_category": "permissive"},
            {"id": "s2", "license_category": "restricted"},
        ]
        report = check_license_compatibility(sources, usage_context="commercial")
        assert not report.passed
        assert report.compatible_sources == 1
        assert report.incompatible_sources == 1
        incompatible = [f for f in report.findings if not f["compatible"]]
        assert len(incompatible) == 1
        assert "commercial" in incompatible[0]["reason"]

    def test_restricted_license_passes_research(self):
        sources = [{"id": "s1", "license_category": "restricted"}]
        report = check_license_compatibility(sources, usage_context="research")
        assert report.passed
        assert report.compatible_sources == 1

    def test_prohibited_license_fails_all(self):
        for context in ("commercial", "research", "redistribution"):
            sources = [{"id": "s1", "license_category": "prohibited"}]
            report = check_license_compatibility(sources, usage_context=context)
            assert not report.passed

    def test_unknown_license_category(self):
        sources = [{"id": "s1", "license_category": "weird_license"}]
        report = check_license_compatibility(sources, usage_context="commercial")
        assert not report.passed
        finding = report.findings[0]
        assert not finding["compatible"]
        assert "unknown license category" in finding["reason"]

    def test_missing_license_category_defaults_to_unknown(self):
        sources = [{"id": "s1"}]
        report = check_license_compatibility(sources, usage_context="commercial")
        assert not report.passed
        finding = report.findings[0]
        assert finding["license_category"] == "unknown"

    def test_rejects_invalid_usage_context(self):
        sources = [{"id": "s1", "license_category": "permissive"}]
        with pytest.raises(ValueError, match="usage_context"):
            check_license_compatibility(sources, usage_context="evil")

    def test_rejects_empty_sources(self):
        with pytest.raises(ValueError, match="not be empty"):
            check_license_compatibility([], usage_context="commercial")

    def test_report_to_dict(self):
        sources = [{"id": "s1", "license_category": "permissive"}]
        report = check_license_compatibility(sources, usage_context="commercial")
        payload = report.to_dict()
        assert payload["schema_version"] == 1
        assert payload["usage_context"] == "commercial"
        assert payload["passed"] is True
        assert len(payload["findings"]) == 1


