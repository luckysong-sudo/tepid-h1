import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DeltaBackendValidationTests(unittest.TestCase):
    def test_eager_compiler_boundary_matches_reference(self):
        from tepid_h1.evaluation.delta_backend import (
            DeltaBackendValidationConfig,
            validate_delta_backend,
        )

        report = validate_delta_backend(
            DeltaBackendValidationConfig(
                sequence_length=3,
                iterations=1,
                seed=79,
                target_device_label="declared-but-not-cuda",
            )
        )

        self.assertTrue(report["numerical_passed"])
        self.assertEqual(report["implementations"]["candidate"], "GatedDeltaMemoryEager")
        self.assertTrue(all(item["passed"] for item in report["comparisons"].values()))
        self.assertFalse(report["qualification"]["target_hardware_evidence"])
        self.assertFalse(report["qualification"]["optimization_qualified"])
        self.assertGreater(report["timing"]["reference_tokens_per_second"], 0)
        self.assertGreater(report["timing"]["candidate_tokens_per_second"], 0)
        self.assertIn("not declared", report["qualification"]["reason"])

    def test_invalid_validation_config_is_rejected(self):
        from tepid_h1.evaluation.delta_backend import DeltaBackendValidationConfig

        with self.assertRaisesRegex(ValueError, "backend"):
            DeltaBackendValidationConfig(backend="unknown")
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            DeltaBackendValidationConfig(sequence_length=1)
        with self.assertRaisesRegex(ValueError, "target_device_label"):
            DeltaBackendValidationConfig(target_device_label=" ")


if __name__ == "__main__":
    unittest.main()
