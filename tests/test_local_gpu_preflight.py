import unittest

try:
    import torch
except ImportError:
    torch = None


class LocalGPUPreflightTests(unittest.TestCase):
    def test_nvidia_smi_query_output_is_parsed(self):
        from tepid_h1.integrations.local_gpu import _parse_nvidia_smi_query

        self.assertEqual(
            _parse_nvidia_smi_query("GeForce MX150, 388.73, 2048\n"),
            [
                {
                    "name": "GeForce MX150",
                    "driver_version": "388.73",
                    "driver_major": 388,
                    "legacy_driver": True,
                    "memory_total_mib": 2048,
                }
            ],
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_missing_nvidia_smi_reports_blocker(self):
        from tepid_h1.integrations import (
            LocalGPUPreflightConfig,
            build_local_gpu_preflight_report,
        )

        report = build_local_gpu_preflight_report(
            LocalGPUPreflightConfig(nvidia_smi_path="Z:/missing/nvidia-smi.exe")
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["experiment"], "local_gpu_preflight")
        self.assertFalse(report["ready_for_cuda"])
        self.assertIn("nvidia-smi did not report", report["blockers"][0])
        self.assertEqual(report["hardware"]["gpus"], [])
        self.assertIn("cuda_available", report["torch"])
        self.assertTrue(report["recommended_actions"])
        self.assertEqual(report["capacity_warnings"], [])
        self.assertEqual(report["readiness"]["cuda_runtime"]["status"], "blocked")
        self.assertEqual(report["validation_plan"][1]["status"], "blocked")

    def test_blockers_include_cuda_torch_action(self):
        from tepid_h1.integrations.local_gpu import _recommended_actions

        actions = _recommended_actions(
            {"gpus": [{"name": "GeForce MX150"}]},
            {"cuda_runtime": None, "cuda_available": False},
            ["installed PyTorch build does not include CUDA"],
        )

        self.assertIn("CUDA-enabled PyTorch", " ".join(actions))
        self.assertIn("rerun gpu-preflight", " ".join(actions))

    def test_legacy_driver_action_is_reported(self):
        from tepid_h1.integrations.local_gpu import _recommended_actions

        actions = _recommended_actions(
            {"gpus": [{"name": "GeForce MX150", "legacy_driver": True}]},
            {"cuda_runtime": None, "cuda_available": False},
            ["installed PyTorch build does not include CUDA"],
        )

        self.assertIn("NVIDIA driver", " ".join(actions))

    def test_validation_plan_marks_cuda_commands_ready_or_blocked(self):
        from tepid_h1.integrations.local_gpu import _validation_plan

        blocked = _validation_plan(ready_for_cuda=False)
        ready = _validation_plan(ready_for_cuda=True)

        self.assertEqual([step["name"] for step in blocked], [step["name"] for step in ready])
        self.assertEqual(blocked[0]["status"], "blocked")
        self.assertEqual(blocked[1]["status"], "blocked")
        self.assertEqual(ready[0]["status"], "passed")
        self.assertEqual(ready[1]["status"], "ready")
        self.assertIn("--device cuda", ready[1]["command"])
        self.assertEqual(ready[1]["scope"], "operator smoke")

    def test_low_memory_gpu_reports_smoke_only_capacity_warning(self):
        from tepid_h1.integrations.local_gpu import _capacity_warnings

        warnings = _capacity_warnings(
            {"gpus": [{"name": "GeForce MX150", "memory_total_mib": 2048}]},
            minimum_operator_memory_mib=4096,
            minimum_scale_training_memory_mib=4096,
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("4096 MiB", warnings[0])
        self.assertIn("smoke checks only", warnings[0])

    def test_low_memory_gpu_reports_scale_training_capacity_warning(self):
        from tepid_h1.integrations.local_gpu import _capacity_warnings

        warnings = _capacity_warnings(
            {"gpus": [{"name": "GeForce MX150", "memory_total_mib": 2048}]},
            minimum_operator_memory_mib=1024,
            minimum_scale_training_memory_mib=24576,
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("24576 MiB", warnings[0])
        self.assertIn("scale-training threshold", warnings[0])

    def test_preflight_config_rejects_invalid_memory_threshold(self):
        from tepid_h1.integrations import LocalGPUPreflightConfig

        with self.assertRaisesRegex(ValueError, "minimum_operator_memory_mib"):
            LocalGPUPreflightConfig(minimum_operator_memory_mib=0)

        with self.assertRaisesRegex(ValueError, "minimum_scale_training_memory_mib"):
            LocalGPUPreflightConfig(minimum_scale_training_memory_mib=0)

        with self.assertRaisesRegex(ValueError, "minimum_scale_training_memory_mib"):
            LocalGPUPreflightConfig(
                minimum_operator_memory_mib=8192,
                minimum_scale_training_memory_mib=4096,
            )

    def test_readiness_distinguishes_smoke_and_scale_training(self):
        from tepid_h1.integrations.local_gpu import _readiness

        blocked = _readiness(
            ["installed PyTorch build does not include CUDA"],
            ["GeForce MX150 reports 2048 MiB VRAM"],
        )
        ready_smoke = _readiness([], ["GeForce MX150 reports 2048 MiB VRAM"])
        not_assessed_scale = _readiness([], [])

        self.assertEqual(blocked["cuda_runtime"]["status"], "blocked")
        self.assertEqual(blocked["operator_smoke"]["status"], "blocked")
        self.assertEqual(ready_smoke["operator_smoke"]["status"], "ready")
        self.assertEqual(ready_smoke["scale_training"]["status"], "blocked")
        self.assertEqual(not_assessed_scale["scale_training"]["status"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
