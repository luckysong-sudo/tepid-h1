"""Tests for the training improvements evidence module."""

import unittest

from tepid_h1.training_improvements import (
    TrainingImprovement,
    count_training_improvements,
    get_training_improvement_ids,
    list_training_improvements,
)


class TrainingImprovementsTests(unittest.TestCase):
    def test_training_improvements_module_exports(self):
        from tepid_h1.training_improvements import (
            TRAINING_IMPROVEMENTS,
            TrainingImprovement,
            get_training_improvement_ids,
            list_training_improvements,
        )

        self.assertTrue(issubclass(TrainingImprovement, object))
        self.assertEqual(count_training_improvements(), 10)
        self.assertIsInstance(get_training_improvement_ids(), tuple)
        self.assertIsInstance(list_training_improvements(), tuple)

    def test_training_improvements_are_frozen(self):
        improvements = list_training_improvements()
        for improvement in improvements:
            self.assertIsInstance(improvement, TrainingImprovement)
            self.assertTrue(
                isinstance(improvement.id, str) and improvement.id.strip(),
                "improvement id must be a non-empty string",
            )
            self.assertTrue(
                isinstance(improvement.category, str) and improvement.category.strip(),
                "improvement category must be a non-empty string",
            )
            self.assertTrue(
                isinstance(improvement.description, str) and improvement.description.strip(),
                "improvement description must be a non-empty string",
            )

    def test_training_improvement_ids_are_unique(self):
        ids = get_training_improvement_ids()
        self.assertEqual(len(ids), len(set(ids)), "improvement IDs must be unique")

    def test_training_improvements_contains_expected_ids(self):
        ids = set(get_training_improvement_ids())
        expected = {
            "training-target-validation",
            "training-eval-label-shape-validation",
            "training-eval-batch-dtype-cardinality-validation",
            "callback-training-empty-epochs",
            "gradient-checkpointing-static-selection",
            "mixed-precision-tensor-dtype-safety",
            "checkpoint-save-invalid-step-validation",
            "checkpoint-load-scheduler-mismatch",
            "checkpoint-load-metadata-rng-validation",
            "paired-smoke-invalid-controls",
        }
        self.assertEqual(ids, expected)

    def test_to_dict_returns_expected_structure(self):
        improvements = list_training_improvements()
        sample = next(
            imp for imp in improvements if imp.id == "paired-smoke-invalid-controls"
        )
        data = sample.to_dict()
        self.assertEqual(
            set(data),
            {"id", "category", "description", "contract_added", "test_coverage", "gap_remaining"},
        )
        self.assertTrue(data["contract_added"])
        self.assertTrue(data["test_coverage"])
        self.assertEqual(data["gap_remaining"], "")

    def test_training_improvements_exported_from_package(self):
        import tepid_h1

        self.assertTrue(hasattr(tepid_h1, "TrainingImprovement"))
        self.assertTrue(hasattr(tepid_h1, "count_training_improvements"))
        self.assertTrue(hasattr(tepid_h1, "get_training_improvement_ids"))
        self.assertTrue(hasattr(tepid_h1, "list_training_improvements"))

    def test_training_improvements_are_consistent_with_top_level_exports(self):
        import tepid_h1

        top_level = set(tepid_h1.__all__)
        expected = {
            "TrainingImprovement",
            "count_training_improvements",
            "get_training_improvement_ids",
            "list_training_improvements",
        }
        self.assertTrue(expected.issubset(top_level))


if __name__ == "__main__":
    unittest.main()


