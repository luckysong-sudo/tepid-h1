from __future__ import annotations

import unittest

from tepid_h1.project_status import DIMENSIONS, WEIGHTS, build_project_status_report


class ProjectStatusTests(unittest.TestCase):
    def test_project_status_percentages_are_valid(self) -> None:
        report = build_project_status_report()

        self.assertEqual(report.schema_version, 1)
        self.assertEqual(report.prototype_overall_percent, 69)
        self.assertEqual(report.formal_training_overall_percent, 38)
        for dimension in report.dimensions:
            self.assertGreaterEqual(dimension.percent, 0)
            self.assertLessEqual(dimension.percent, 100)
            self.assertTrue(dimension.evidence)
            self.assertTrue(dimension.gaps)

    def test_every_dimension_has_a_weight(self) -> None:
        self.assertEqual({dimension.name for dimension in DIMENSIONS}, set(WEIGHTS))
        self.assertEqual(sum(WEIGHTS.values()), 100)


if __name__ == "__main__":
    unittest.main()
