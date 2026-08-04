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

    def test_benchmark_matrix_reports_shape_level_timing(self):
        from tepid_h1.evaluation.delta_backend import (
            DeltaBackendBenchmarkConfig,
            benchmark_delta_backend,
        )

        report = benchmark_delta_backend(
            DeltaBackendBenchmarkConfig(
                sequence_lengths=(2, 3),
                iterations=1,
                seed=83,
            )
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["experiment"], "delta_backend_benchmark_matrix")
        self.assertEqual(report["environment"]["device"], "cpu")
        self.assertEqual(report["environment"]["dtype"], "float32")
        self.assertFalse(report["environment"]["target_device_label_declared"])
        self.assertEqual(report["environment"]["sequence_length_min"], 2)
        self.assertEqual(report["environment"]["sequence_length_max"], 3)
        self.assertEqual(report["summary"]["case_count"], 2)
        self.assertTrue(report["summary"]["all_numerical_passed"])
        self.assertFalse(report["summary"]["all_optimization_qualified"])
        self.assertEqual(report["summary"]["qualified_case_count"], 0)
        self.assertEqual(report["summary"]["target_hardware_case_count"], 0)
        self.assertEqual(
            report["summary"]["qualification_reasons"],
            {"target CUDA device with the inductor backend was not declared": 2},
        )
        self.assertEqual([case["sequence_length"] for case in report["cases"]], [2, 3])
        self.assertEqual([case["shape_role"] for case in report["cases"]], ["minimum", "maximum"])
        for case in report["cases"]:
            self.assertRegex(case["case_id"], r"^delta-eager-cpu-float32-b1-s[23]$")
            self.assertEqual(case["batch_size"], 1)
            self.assertEqual(case["device"], "cpu")
            self.assertEqual(case["dtype"], "float32")
            self.assertIsNone(case["target_device_label"])
            self.assertFalse(case["target_hardware_evidence"])
            self.assertGreater(case["tokens"], 0)
            self.assertGreater(case["reference_tokens_per_second"], 0)
            self.assertGreater(case["candidate_tokens_per_second"], 0)
            self.assertIn("not declared", case["qualification_reason"])

    def test_benchmark_matrix_marks_single_shape_role(self):
        from tepid_h1.evaluation.delta_backend import (
            DeltaBackendBenchmarkConfig,
            benchmark_delta_backend,
        )

        report = benchmark_delta_backend(
            DeltaBackendBenchmarkConfig(
                sequence_lengths=(4,),
                iterations=1,
                seed=89,
            )
        )

        self.assertEqual(report["cases"][0]["shape_role"], "single")

    def test_invalid_benchmark_config_is_rejected(self):
        from tepid_h1.evaluation.delta_backend import DeltaBackendBenchmarkConfig

        with self.assertRaisesRegex(ValueError, "sequence_lengths"):
            DeltaBackendBenchmarkConfig(sequence_lengths=())
        with self.assertRaisesRegex(ValueError, "sequence_lengths"):
            DeltaBackendBenchmarkConfig(sequence_lengths=(1,))


if __name__ == "__main__":
    unittest.main()
