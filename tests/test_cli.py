import json
import tempfile
import unittest
from pathlib import Path


class CLIParserTests(unittest.TestCase):
    def test_plan_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["plan"])
        self.assertEqual(args.command, "plan")
        self.assertEqual(args.variant, "prototype")

        args_reference = parser.parse_args(["plan", "--variant", "reference"])
        self.assertEqual(args_reference.variant, "reference")

    def test_data_audit_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            args = parser.parse_args(["data-audit", f.name])
            self.assertEqual(args.command, "data-audit")
            self.assertEqual(Path(args.inventory), Path(f.name))

    def test_decontaminate_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as training, tempfile.NamedTemporaryFile(
            suffix=".jsonl"
        ) as benchmark:
            args = parser.parse_args(
                [
                    "decontaminate",
                    "--training",
                    training.name,
                    "--benchmark",
                    benchmark.name,
                ]
            )
            self.assertEqual(args.ngram_size, 5)
            self.assertAlmostEqual(args.threshold, 0.8, places=6)

    def test_tokenizer_benchmark_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as corpus:
            args = parser.parse_args(
                [
                    "tokenizer-benchmark",
                    "--corpus",
                    corpus.name,
                    "--candidate",
                    "64000=/fake/64k.json",
                    "--candidate",
                    "80000=/fake/80k.json",
                    "--candidate",
                    "96000=/fake/96k.json",
                ]
            )
            self.assertEqual(len(args.candidate), 3)

    def test_train_smoke_command_validation(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["train-smoke"])
        self.assertEqual(args.command, "train-smoke")

    def test_corpus_stats_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as corpus:
            args = parser.parse_args(["corpus-stats", corpus.name])
            self.assertEqual(args.command, "corpus-stats")

    def test_compare_smoke_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "compare-smoke",
                "--steps",
                "1",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(args.steps, 1)
        self.assertEqual(args.device, "cpu")


class CLIIntegrationTests(unittest.TestCase):
    def test_plan_command_outputs_json(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        with patch.object(sys, "argv", ["tepid-h1", "plan"]), patch(
            "sys.stdout", new=StringIO()
        ) as mock_stdout:
            result = main()
            self.assertEqual(result, 0)
            output = mock_stdout.getvalue()
            plan = json.loads(output)

        self.assertIn("config", plan)
        self.assertIn("module_counts", plan)
        self.assertIn("layers", plan)
        self.assertEqual(plan["config"]["vocab_size"], 4096)
        self.assertEqual(plan["config"]["num_layers"], 8)

    def test_data_audit_requires_inventory(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["data-audit"])

    def test_moe_balance_report_arguments(self):
        from tepid_h1.cli import build_parser

        args = build_parser().parse_args(
            ["moe-balance-report", "--batch-size", "2", "--steps", "3", "--max-load-cv", "0.3"]
        )

        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.steps, 3)
        self.assertEqual(args.max_load_cv, 0.3)

    def test_moe_balance_report_produces_valid_json(self):
        import json
        import sys
        import tempfile
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report_path = f.name
        try:
            with patch.object(sys, "argv", [
                "tepid-h1", "moe-balance-report",
                "--batch-size", "2", "--sequence-length", "8",
                "--steps", "1", "--seed", "42",
                "--max-load-cv", "0.5",
                "--report", report_path,
            ]):
                from tepid_h1.cli import main
                result = main()

            self.assertEqual(result, 0)
            data = json.loads(Path(report_path).read_text())
            self.assertEqual(data["schema_version"], 1)
            self.assertTrue(data["routing"]["passed"])
            self.assertEqual(data["routing"]["moe_layers"], 2)
            self.assertEqual(data["config"]["steps"], 1)
        finally:
            Path(report_path).unlink(missing_ok=True)

    def test_delta_validate_skip_gradients_argument(self):
        from tepid_h1.cli import build_parser

        args = build_parser().parse_args([
            "delta-validate", "--backend", "eager",
            "--skip-gradients", "--sequence-length", "16",
        ])
        self.assertTrue(args.skip_gradients)

    def test_train_smoke_command_accepts_valid_steps(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["train-smoke", "--steps", "2"])
        self.assertEqual(args.steps, 2)

    def test_train_smoke_command_accepts_valid_seq_len(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["train-smoke", "--sequence-length", "16"])
        self.assertEqual(args.sequence_length, 16)

    def test_compare_smoke_command_accepts_corpus_arg(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as corpus:
            args = parser.parse_args(["compare-smoke", "--corpus", corpus.name])
            self.assertEqual(str(args.corpus), corpus.name)

    def test_corpus_stats_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as corpus:
            args = parser.parse_args(["corpus-stats", corpus.name])
            self.assertEqual(args.command, "corpus-stats")

    def test_corpus_compare_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as training, tempfile.NamedTemporaryFile(suffix=".jsonl") as validation:
            args = parser.parse_args(["corpus-compare", training.name, validation.name])
            self.assertEqual(args.command, "corpus-compare")

    def test_delta_validate_skip_gradients_produces_report(self):
        import json
        import sys
        import tempfile
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report_path = f.name
        try:
            with patch.object(sys, "argv", [
                "tepid-h1", "delta-validate",
                "--backend", "eager", "--device", "cpu",
                "--skip-gradients", "--sequence-length", "8",
                "--iterations", "1", "--seed", "71",
                "--report", report_path,
            ]):
                from tepid_h1.cli import main
                result = main()

            self.assertEqual(result, 0)
            data = json.loads(Path(report_path).read_text())
            self.assertTrue(data["numerical_passed"])
            self.assertEqual(data["config"]["verify_gradients"], False)
        finally:
            Path(report_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
