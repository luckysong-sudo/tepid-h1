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
        self.assertEqual(report["summary"]["case_count"], 2)
        self.assertTrue(report["summary"]["all_numerical_passed"])
        self.assertGreater(report["summary"]["min_grouped_over_dispatch_speedup"], 0)
        self.assertEqual([case["sequence_length"] for case in report["cases"]], [2, 3])
        self.assertEqual(report["model"]["top_k"], 2)
        for case in report["cases"]:
            router = case["router"]
            self.assertTrue(case["numerical_passed"])
            self.assertLessEqual(case["max_abs_error"], 1e-6)
            self.assertGreater(case["dispatch_oracle_tokens_per_second"], 0)
            self.assertGreater(case["grouped_tokens_per_second"], 0)
            self.assertGreater(case["grouped_over_dispatch_speedup"], 0)
            self.assertGreater(case["tokens_per_second"], 0)
            self.assertEqual(case["tokens_per_second"], case["grouped_tokens_per_second"])
            self.assertEqual(router["total_assignments"], router["expected_assignments"])
            self.assertEqual(router["expected_assignments"], case["batch_tokens"] * 2)
            self.assertEqual(case["tokens"], case["batch_tokens"] * 2)
            self.assertGreaterEqual(router["active_experts"], 1)
            self.assertGreater(router["mean_router_entropy"], 0)

    def test_invalid_benchmark_config_is_rejected(self):
        from tepid_h1.evaluation.moe_backend import RoutedMoEBenchmarkConfig

        with self.assertRaisesRegex(ValueError, "variant"):
            RoutedMoEBenchmarkConfig(variant="reference")
        with self.assertRaisesRegex(ValueError, "sequence_lengths"):
            RoutedMoEBenchmarkConfig(sequence_lengths=())
        with self.assertRaisesRegex(ValueError, "sequence_lengths"):
            RoutedMoEBenchmarkConfig(sequence_lengths=(0,))


if __name__ == "__main__":
    unittest.main()
