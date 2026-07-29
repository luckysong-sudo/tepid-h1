import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ZeroGPUAdapterTests(unittest.TestCase):
    def test_cpu_injected_job_preserves_service_limits_and_governance(self):
        from tepid_h1.integrations import ZeroGPUJobConfig, run_zero_gpu_job

        report = run_zero_gpu_job(
            ZeroGPUJobConfig(steps=1, trials=1, dtype="float32"),
            corpus_path=ROOT / "configs" / "paired_corpus.example.jsonl",
            inventory_path=ROOT / "configs" / "data_inventory.example.json",
            device="cpu",
            core_revision="test-revision",
        )

        self.assertEqual(report["environment"]["device_type"], "cpu")
        self.assertEqual(report["data"]["source_id"], "synthetic-fixture-v1")
        self.assertEqual(
            report["deployment_adapter"]["name"],
            "huggingface_zerogpu_gradio",
        )
        self.assertEqual(report["deployment_adapter"]["core_revision"], "test-revision")
        self.assertEqual(report["deployment_adapter"]["limits"]["maximum_trials"], 3)

    def test_job_limits_fail_closed(self):
        from tepid_h1.integrations import ZeroGPUJobConfig

        with self.assertRaisesRegex(ValueError, "steps"):
            ZeroGPUJobConfig(steps=6)
        with self.assertRaisesRegex(ValueError, "trials"):
            ZeroGPUJobConfig(trials=4)
        with self.assertRaisesRegex(ValueError, "dtype"):
            ZeroGPUJobConfig(dtype="float64")
        with self.assertRaisesRegex(TypeError, "integer"):
            ZeroGPUJobConfig(steps=1.5)


if __name__ == "__main__":
    unittest.main()
