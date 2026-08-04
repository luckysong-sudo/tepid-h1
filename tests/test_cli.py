import argparse
import json
import tempfile
import unittest
from pathlib import Path


EXPECTED_SUBCOMMANDS = [
    "baseline-report",
    "compare-smoke",
    "corpus-compare",
    "corpus-stats",
    "data-audit",
    "decontaminate",
    "delta-benchmark",
    "delta-validate",
    "gpu-preflight",
    "moe-benchmark",
    "plan",
    "project-status",
    "retrieval-generate",
    "retrieval-score",
    "stage-gates",
    "tokenizer-benchmark",
    "train-smoke",
    "training-improve",
]


def _subcommands(parser: argparse.ArgumentParser) -> list[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    raise AssertionError("parser must define subcommands")


class CLIParserTests(unittest.TestCase):
    def test_subcommand_inventory_is_stable(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        self.assertEqual(_subcommands(parser), EXPECTED_SUBCOMMANDS)

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

    def test_stage_gates_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["stage-gates"])
        self.assertEqual(args.command, "stage-gates")
        self.assertEqual(Path(args.config), Path("configs/stage_gates.json"))

    def test_project_status_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["project-status"])
        self.assertEqual(args.command, "project-status")

    def test_gpu_preflight_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "gpu-preflight",
                "--nvidia-smi",
                "C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe",
                "--minimum-operator-memory-mib",
                "4096",
                "--minimum-scale-training-memory-mib",
                "24576",
            ]
        )
        self.assertEqual(args.command, "gpu-preflight")
        self.assertEqual(args.nvidia_smi.name, "nvidia-smi.exe")
        self.assertEqual(args.minimum_operator_memory_mib, 4096)
        self.assertEqual(args.minimum_scale_training_memory_mib, 24576)

    def test_decontaminate_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with (
            tempfile.NamedTemporaryFile(suffix=".jsonl") as training,
            tempfile.NamedTemporaryFile(suffix=".jsonl") as benchmark,
        ):
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

    def test_delta_benchmark_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "delta-benchmark",
                "--length",
                "2",
                "--iterations",
                "1",
            ]
        )
        self.assertEqual(args.command, "delta-benchmark")
        self.assertEqual(args.sequence_lengths, [2])
        self.assertEqual(args.iterations, 1)

    def test_moe_benchmark_command(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "moe-benchmark",
                "--variant",
                "prototype",
                "--length",
                "2",
                "--iterations",
                "1",
                "--target-device-label",
                "local-gpu",
                "--router-assignment-cv-threshold",
                "0.5",
                "--minimum-grouped-speedup",
                "1.25",
                "--require-m4-proxy",
            ]
        )
        self.assertEqual(args.command, "moe-benchmark")
        self.assertEqual(args.variant, "prototype")
        self.assertEqual(args.sequence_lengths, [2])
        self.assertEqual(args.iterations, 1)
        self.assertEqual(args.target_device_label, "local-gpu")
        self.assertEqual(args.router_assignment_cv_threshold, 0.5)
        self.assertEqual(args.minimum_grouped_speedup, 1.25)
        self.assertTrue(args.require_m4_proxy)


class CLIIntegrationTests(unittest.TestCase):
    def test_plan_command_outputs_json(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        with patch.object(sys, "argv", ["tepid-h1", "plan"]):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                result = main()
                self.assertEqual(result, 0)
                output = mock_stdout.getvalue()
                plan = json.loads(output)

        self.assertIn("config", plan)
        self.assertIn("module_counts", plan)
        self.assertIn("layers", plan)
        self.assertEqual(set(plan), {"config", "module_counts", "layers"})
        self.assertEqual(plan["config"]["vocab_size"], 4096)
        self.assertEqual(plan["config"]["num_layers"], 8)

    def test_data_audit_requires_inventory(self):
        from tepid_h1.cli import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["data-audit"])

    def test_stage_gates_command_outputs_json(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        with patch.object(sys, "argv", ["tepid-h1", "stage-gates"]):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                result = main()
                self.assertEqual(result, 0)
                output = mock_stdout.getvalue()
                report = json.loads(output)

        self.assertEqual(set(report), {"schema_version", "passed", "gates", "errors"})
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["gates"]), 6)
        self.assertEqual(
            set(report["gates"][0]),
            {"name", "deliverables", "exit_criteria", "evidence_refs"},
        )
        self.assertTrue(report["gates"][4]["evidence_refs"])

    def test_stage_gates_command_rejects_unknown_cli_evidence_ref(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        payload = {
            "M0_data_tokenizer": {
                "deliverables": ["deliverable"],
                "exit_criteria": ["criterion"],
                "evidence_refs": ["cli:tepid-h1 unknown-command"],
            },
            "M1_350m_prototype": {
                "deliverables": ["deliverable"],
                "exit_criteria": ["criterion"],
                "evidence_refs": ["cli:tepid-h1 plan"],
            },
            "M2_1_3b_ablation": {
                "deliverables": ["deliverable"],
                "exit_criteria": ["criterion"],
                "evidence_refs": ["cli:tepid-h1 project-status"],
            },
            "M3_7b_product_evidence": {
                "deliverables": ["deliverable"],
                "exit_criteria": ["criterion"],
                "evidence_refs": ["cli:tepid-h1 stage-gates"],
            },
            "M4_moe_prototype": {
                "deliverables": ["deliverable"],
                "exit_criteria": ["criterion"],
                "evidence_refs": ["cli:tepid-h1 moe-benchmark --length 2 --iterations 1"],
            },
            "M5_formal_training": {
                "deliverables": ["deliverable"],
                "exit_criteria": ["criterion"],
                "evidence_refs": ["cli:tepid-h1 stage-gates"],
            },
        }
        with tempfile.TemporaryDirectory() as tempdir:
            config = Path(tempdir) / "stage_gates.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(sys, "argv", ["tepid-h1", "stage-gates", str(config)]):
                with patch("sys.stdout", new=StringIO()) as mock_stdout:
                    result = main()
                    self.assertEqual(result, 8)
                    report = json.loads(mock_stdout.getvalue())

        self.assertFalse(report["passed"])
        self.assertTrue(any("invalid CLI ref" in error for error in report["errors"]))

    def test_project_status_command_outputs_json(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        with patch.object(sys, "argv", ["tepid-h1", "project-status"]):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                result = main()
                self.assertEqual(result, 0)
                output = mock_stdout.getvalue()
                report = json.loads(output)

        self.assertEqual(
            set(report),
            {
                "schema_version",
                "prototype_scope",
                "prototype_overall_percent",
                "formal_training_overall_percent",
                "dimensions",
                "interpretation",
            },
        )
        self.assertEqual(report["prototype_overall_percent"], 81)
        self.assertEqual(report["formal_training_overall_percent"], 38)
        self.assertEqual(len(report["dimensions"]), 9)
        self.assertEqual(
            set(report["dimensions"][0]),
            {"name", "percent", "evidence", "gaps"},
        )

    def test_gpu_preflight_not_ready_outputs_json_and_nonzero_status(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        preflight_report = {
            "schema_version": 1,
            "experiment": "local_gpu_preflight",
            "config": {"nvidia_smi_path": None},
            "hardware": {"gpus": []},
            "torch": {"cuda_available": False},
            "ready_for_cuda": False,
            "blockers": ["installed PyTorch build does not include CUDA"],
            "capacity_warnings": [],
            "readiness": {"cuda_runtime": {"status": "blocked"}},
            "recommended_actions": ["install a CUDA-enabled PyTorch build"],
            "validation_plan": [{"name": "delta_cuda_benchmark", "status": "blocked"}],
            "interpretation": "not ready",
        }
        with patch.object(sys, "argv", ["tepid-h1", "gpu-preflight"]):
            with patch(
                "tepid_h1.integrations.build_local_gpu_preflight_report",
                return_value=preflight_report,
            ):
                with patch("sys.stdout", new=StringIO()) as mock_stdout:
                    result = main()
                    output = json.loads(mock_stdout.getvalue())

        self.assertEqual(result, 10)
        self.assertEqual(
            set(output),
            {
                "schema_version",
                "experiment",
                "config",
                "hardware",
                "torch",
                "ready_for_cuda",
                "blockers",
                "capacity_warnings",
                "readiness",
                "recommended_actions",
                "validation_plan",
                "interpretation",
            },
        )
        self.assertFalse(output["ready_for_cuda"])
        self.assertTrue(output["blockers"])

    def test_gpu_preflight_ready_returns_zero_status(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        preflight_report = {
            "schema_version": 1,
            "experiment": "local_gpu_preflight",
            "config": {"nvidia_smi_path": None},
            "hardware": {"gpus": [{"name": "CUDA GPU"}]},
            "torch": {"cuda_available": True},
            "ready_for_cuda": True,
            "blockers": [],
            "capacity_warnings": [],
            "readiness": {"cuda_runtime": {"status": "ready"}},
            "recommended_actions": ["run tepid-h1 delta-benchmark --device cuda"],
            "validation_plan": [{"name": "delta_cuda_benchmark", "status": "ready"}],
            "interpretation": "ready",
        }
        with patch.object(sys, "argv", ["tepid-h1", "gpu-preflight"]):
            with patch(
                "tepid_h1.integrations.build_local_gpu_preflight_report",
                return_value=preflight_report,
            ):
                with patch("sys.stdout", new=StringIO()):
                    result = main()

        self.assertEqual(result, 0)

    def test_moe_benchmark_report_mode_returns_zero_when_proxy_blocked(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        report = {
            "schema_version": 1,
            "summary": {"m4_moe_proxy_passed": False},
        }
        with patch.object(sys, "argv", ["tepid-h1", "moe-benchmark"]):
            with patch(
                "tepid_h1.evaluation.moe_backend.benchmark_routed_moe",
                return_value=report,
            ):
                with patch("sys.stdout", new=StringIO()) as mock_stdout:
                    result = main()
                    output = json.loads(mock_stdout.getvalue())

        self.assertEqual(result, 0)
        self.assertFalse(output["summary"]["m4_moe_proxy_passed"])

    def test_training_improve_command(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        with patch.object(sys, "argv", ["tepid-h1", "training-improve"]):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                result = main()
                output = json.loads(mock_stdout.getvalue())

        self.assertEqual(result, 0)
        self.assertEqual(output["schema_version"], 1)
        self.assertEqual(output["category_filter"], None)
        self.assertGreater(output["count"], 0)
        self.assertIsInstance(output["improvements"], list)

    def test_training_improve_command_with_category_filter(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        with patch.object(sys, "argv", ["tepid-h1", "training-improve", "--category", "training"]):
            with patch("sys.stdout", new=StringIO()) as mock_stdout:
                result = main()
                output = json.loads(mock_stdout.getvalue())

        self.assertEqual(result, 0)
        self.assertEqual(output["category_filter"], "training")
        self.assertGreater(output["count"], 0)
        for record in output["improvements"]:
            self.assertEqual(record["category"], "training")

    def test_moe_benchmark_require_m4_proxy_returns_nonzero_when_blocked(self):
        import sys
        from io import StringIO
        from unittest.mock import patch

        from tepid_h1.cli import main

        report = {
            "schema_version": 1,
            "summary": {"m4_moe_proxy_passed": False},
        }
        with patch.object(sys, "argv", ["tepid-h1", "moe-benchmark", "--require-m4-proxy"]):
            with patch(
                "tepid_h1.evaluation.moe_backend.benchmark_routed_moe",
                return_value=report,
            ):
                with patch("sys.stdout", new=StringIO()) as mock_stdout:
                    result = main()
                    output = json.loads(mock_stdout.getvalue())

        self.assertEqual(result, 11)
        self.assertFalse(output["summary"]["m4_moe_proxy_passed"])


if __name__ == "__main__":
    unittest.main()
