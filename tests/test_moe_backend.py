import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RoutedMoEBenchmarkTests(unittest.TestCase):
    def test_benchmark_reports_routing_load_and_timing(self):
        from tepid_h1.evaluation.moe_backend import (
            RoutedMoEBenchmarkConfig,
            benchmark_routed_moe,
        )

        report = benchmark_routed_moe(
            RoutedMoEBenchmarkConfig(
                sequence_lengths=(2, 3),
                iterations=2,
                seed=101,
            )
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["experiment"], "routed_moe_benchmark_matrix")
        self.assertEqual(report["environment"]["device_type"], "cpu")
        self.assertEqual(report["environment"]["dtype"], "float32")
        self.assertEqual(report["environment"]["cuda_available"], False)
        self.assertFalse(report["environment"]["target_device_label_declared"])
        self.assertEqual(report["environment"]["sequence_length_min"], 2)
        self.assertEqual(report["environment"]["sequence_length_max"], 3)
        self.assertEqual(report["summary"]["case_count"], 2)
        self.assertTrue(report["summary"]["all_numerical_passed"])
        self.assertEqual(report["summary"]["target_hardware_case_count"], 0)
        self.assertEqual(report["summary"]["shape_roles"], {"minimum": 1, "maximum": 1})
        self.assertFalse(report["summary"]["m4_moe_proxy_passed"])
        self.assertEqual(report["summary"]["m4_moe_proxy_status"], "blocked")
        self.assertTrue(report["summary"]["m4_moe_proxy_blockers"])
        self.assertIn("target-hardware", " ".join(report["summary"]["m4_moe_proxy_blockers"]))
        self.assertGreater(report["summary"]["min_grouped_over_dispatch_speedup"], 0)
        self.assertEqual(report["summary"]["minimum_grouped_over_dispatch_speedup"], 1.0)
        self.assertIs(
            report["summary"]["all_grouped_speedups_meet_threshold"],
            all(case["grouped_speedup_meets_threshold"] for case in report["cases"]),
        )
        self.assertIn(report["summary"]["grouped_speedup_status"], {"passed", "failed"})
        self.assertIn("threshold", report["summary"]["grouped_speedup_reason"])
        self.assertGreaterEqual(report["summary"]["min_router_assignment_cv"], 0)
        self.assertGreaterEqual(
            report["summary"]["max_router_assignment_cv"],
            report["summary"]["min_router_assignment_cv"],
        )
        self.assertEqual(report["summary"]["router_assignment_cv_threshold"], 0.25)
        self.assertIs(
            report["summary"]["all_router_assignment_cv_within_threshold"],
            all(
                case["router"]["assignment_cv_within_threshold"]
                for case in report["cases"]
            ),
        )
        self.assertIn(
            report["summary"]["router_assignment_cv_status"],
            {"passed", "failed"},
        )
        self.assertIn("threshold", report["summary"]["router_assignment_cv_reason"])
        worst_case = max(
            report["cases"],
            key=lambda case: case["router"]["assignment_coefficient_of_variation"],
        )
        self.assertEqual(
            report["summary"]["max_router_assignment_cv_case_id"],
            worst_case["case_id"],
        )
        self.assertEqual(
            report["summary"]["max_router_assignment_cv_sequence_length"],
            worst_case["sequence_length"],
        )
        slowest_case = min(
            report["cases"],
            key=lambda case: case["grouped_over_dispatch_speedup"],
        )
        self.assertEqual(
            report["summary"]["min_grouped_over_dispatch_speedup_case_id"],
            slowest_case["case_id"],
        )
        self.assertEqual(
            report["summary"]["min_grouped_over_dispatch_speedup_sequence_length"],
            slowest_case["sequence_length"],
        )
        self.assertEqual([case["sequence_length"] for case in report["cases"]], [2, 3])
        self.assertEqual([case["shape_role"] for case in report["cases"]], ["minimum", "maximum"])
        self.assertEqual(report["model"]["top_k"], 2)
        for case in report["cases"]:
            router = case["router"]
            self.assertRegex(case["case_id"], r"^moe-smoke-cpu-float32-b1-s[23]$")
            self.assertEqual(case["batch_size"], 1)
            self.assertEqual(case["device"], "cpu")
            self.assertEqual(case["dtype"], "float32")
            self.assertIsNone(case["target_device_label"])
            self.assertFalse(case["target_hardware_evidence"])
            self.assertTrue(case["numerical_passed"])
            self.assertLessEqual(case["max_abs_error"], 1e-6)
            self.assertGreater(case["dispatch_oracle_tokens_per_second"], 0)
            self.assertGreater(case["grouped_tokens_per_second"], 0)
            self.assertGreater(case["grouped_over_dispatch_speedup"], 0)
            self.assertEqual(case["minimum_grouped_over_dispatch_speedup"], 1.0)
            self.assertEqual(
                case["grouped_speedup_meets_threshold"],
                case["grouped_over_dispatch_speedup"] >= 1.0,
            )
            self.assertIn(case["grouped_speedup_status"], {"passed", "failed"})
            self.assertIn("threshold", case["grouped_speedup_reason"])
            self.assertGreater(case["tokens_per_second"], 0)
            self.assertEqual(case["tokens_per_second"], case["grouped_tokens_per_second"])
            self.assertEqual(router["total_assignments"], router["expected_assignments"])
            self.assertEqual(router["expected_assignments"], case["batch_tokens"] * 2)
            self.assertEqual(case["tokens"], case["batch_tokens"] * 2)
            self.assertGreaterEqual(router["active_experts"], 1)
            self.assertGreaterEqual(router["assignment_coefficient_of_variation"], 0)
            self.assertEqual(router["assignment_cv_threshold"], 0.25)
            self.assertEqual(
                router["assignment_cv_within_threshold"],
                router["assignment_coefficient_of_variation"] <= 0.25,
            )
            self.assertIn(router["assignment_cv_status"], {"passed", "failed"})
            self.assertIn("threshold", router["assignment_cv_reason"])
            self.assertGreater(router["mean_router_entropy"], 0)

    def test_invalid_benchmark_config_is_rejected(self):
        from tepid_h1.evaluation.moe_backend import RoutedMoEBenchmarkConfig

        with self.assertRaisesRegex(ValueError, "variant"):
            RoutedMoEBenchmarkConfig(variant="reference")
        with self.assertRaisesRegex(ValueError, "sequence_lengths"):
            RoutedMoEBenchmarkConfig(sequence_lengths=())
        with self.assertRaisesRegex(ValueError, "sequence_lengths"):
            RoutedMoEBenchmarkConfig(sequence_lengths=(0,))
        with self.assertRaisesRegex(TypeError, "batch_size"):
            RoutedMoEBenchmarkConfig(batch_size=True)
        with self.assertRaisesRegex(TypeError, "iterations"):
            RoutedMoEBenchmarkConfig(iterations=False)
        with self.assertRaisesRegex(TypeError, "seed"):
            RoutedMoEBenchmarkConfig(seed="97")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "sequence_lengths"):
            RoutedMoEBenchmarkConfig(sequence_lengths=[2])  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "sequence_lengths"):
            RoutedMoEBenchmarkConfig(sequence_lengths=(True,))
        with self.assertRaisesRegex(ValueError, "target_device_label"):
            RoutedMoEBenchmarkConfig(target_device_label=" ")
        with self.assertRaisesRegex(ValueError, "router_assignment_cv_threshold"):
            RoutedMoEBenchmarkConfig(router_assignment_cv_threshold=-0.1)
        with self.assertRaisesRegex(ValueError, "router_assignment_cv_threshold"):
            RoutedMoEBenchmarkConfig(router_assignment_cv_threshold=True)
        with self.assertRaisesRegex(ValueError, "router_assignment_cv_threshold"):
            RoutedMoEBenchmarkConfig(router_assignment_cv_threshold=float("nan"))
        with self.assertRaisesRegex(ValueError, "minimum_grouped_over_dispatch_speedup"):
            RoutedMoEBenchmarkConfig(minimum_grouped_over_dispatch_speedup=-0.1)
        with self.assertRaisesRegex(ValueError, "minimum_grouped_over_dispatch_speedup"):
            RoutedMoEBenchmarkConfig(minimum_grouped_over_dispatch_speedup=True)
        with self.assertRaisesRegex(ValueError, "minimum_grouped_over_dispatch_speedup"):
            RoutedMoEBenchmarkConfig(minimum_grouped_over_dispatch_speedup=float("inf"))

    def test_benchmark_matrix_marks_single_shape_role(self):
        from tepid_h1.evaluation.moe_backend import RoutedMoEBenchmarkConfig, benchmark_routed_moe

        report = benchmark_routed_moe(
            RoutedMoEBenchmarkConfig(
                sequence_lengths=(4,),
                iterations=1,
                seed=103,
                target_device_label="declared-but-not-cuda",
                router_assignment_cv_threshold=1.0,
                minimum_grouped_over_dispatch_speedup=0.0,
            )
        )

        self.assertEqual(report["cases"][0]["shape_role"], "single")
        self.assertEqual(report["cases"][0]["target_device_label"], "declared-but-not-cuda")
        self.assertFalse(report["cases"][0]["target_hardware_evidence"])
        self.assertTrue(report["environment"]["target_device_label_declared"])
        self.assertEqual(report["summary"]["router_assignment_cv_threshold"], 1.0)
        self.assertTrue(report["summary"]["all_router_assignment_cv_within_threshold"])
        self.assertEqual(report["summary"]["router_assignment_cv_status"], "passed")
        self.assertEqual(report["cases"][0]["router"]["assignment_cv_status"], "passed")
        self.assertTrue(report["summary"]["all_grouped_speedups_meet_threshold"])
        self.assertEqual(report["summary"]["grouped_speedup_status"], "passed")
        self.assertEqual(report["cases"][0]["grouped_speedup_status"], "passed")
        self.assertFalse(report["summary"]["m4_moe_proxy_passed"])
        self.assertIn("target-hardware", " ".join(report["summary"]["m4_moe_proxy_blockers"]))


if __name__ == "__main__":
    unittest.main()
