import unittest

from tepid_h1.data import audit_inventory


def valid_inventory():
    return {
        "schema_version": 1,
        "inventory_id": "test",
        "owner": "data",
        "generated_at": "2026-07-29T00:00:00Z",
        "sources": [
            {
                "id": "source-1",
                "name": "fixture",
                "uri": "repo://fixture",
                "snapshot": "v1",
                "sha256": "a" * 64,
                "license_id": "CC0-1.0",
                "license_category": "public_domain",
                "commercial_use": True,
                "rights_evidence": "evidence.md",
                "languages": ["zh", "en"],
                "domains": ["text"],
                "estimated_tokens": 123,
                "pii_status": "absent",
                "quality_status": "accepted",
            }
        ],
        "repository_decontamination": {
            "status": "complete",
            "method": "exact and near duplicate",
            "report_uri": "report.md",
            "completed_at": "2026-07-29T00:00:00Z",
            "benchmark_sets": ["heldout"],
        },
    }


class DataAuditTests(unittest.TestCase):
    def test_valid_inventory_passes(self):
        report = audit_inventory(valid_inventory())
        self.assertTrue(report.passed)
        self.assertEqual(report.approved_source_count, 1)
        self.assertEqual(report.estimated_tokens, 123)

    def test_unknown_rights_and_unassessed_pii_fail_closed(self):
        inventory = valid_inventory()
        source = inventory["sources"][0]
        source["license_category"] = "unknown"
        source["commercial_use"] = False
        source["pii_status"] = "unassessed"

        report = audit_inventory(inventory)
        codes = {finding.code for finding in report.findings}
        self.assertFalse(report.passed)
        self.assertEqual(report.approved_source_count, 0)
        self.assertTrue({"license_not_approved", "commercial_use", "pii_not_cleared"} <= codes)

    def test_duplicate_ids_and_incomplete_decontamination_fail(self):
        inventory = valid_inventory()
        inventory["sources"].append(dict(inventory["sources"][0]))
        inventory["repository_decontamination"]["status"] = "planned"

        report = audit_inventory(inventory)
        codes = {finding.code for finding in report.findings}
        self.assertTrue({"duplicate_source_id", "decontamination_status"} <= codes)


if __name__ == "__main__":
    unittest.main()
