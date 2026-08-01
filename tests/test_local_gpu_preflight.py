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


if __name__ == "__main__":
    unittest.main()
