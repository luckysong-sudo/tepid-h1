import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PairedExperimentTests(unittest.TestCase):
    def test_models_receive_identical_token_budget(self):
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        report = run_paired_smoke(
            PairedExperimentConfig(steps=1, batch_size=1, sequence_length=6, seed=43)
        )

        self.assertEqual(report["hybrid"]["trained_tokens"], 5)
        self.assertEqual(report["baseline"]["trained_tokens"], 5)
        self.assertEqual(report["data"]["tokens_per_model"], 5)
        self.assertTrue(report["hybrid"]["parameter_estimate_matches_actual"])
        self.assertTrue(report["baseline"]["parameter_estimate_matches_actual"])
        self.assertGreater(report["hybrid"]["tokens_per_second"], 0)
        self.assertGreater(report["baseline"]["tokens_per_second"], 0)

    def test_same_seed_reproduces_data_and_loss(self):
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(steps=1, sequence_length=5, seed=47)
        first = run_paired_smoke(config)
        second = run_paired_smoke(config)

        self.assertEqual(first["data"]["sha256"], second["data"]["sha256"])
        self.assertEqual(first["hybrid"]["initial_loss"], second["hybrid"]["initial_loss"])
        self.assertEqual(first["baseline"]["initial_loss"], second["baseline"]["initial_loss"])

    def test_invalid_experiment_config_is_rejected(self):
        from tepid_h1.experiments import PairedExperimentConfig

        with self.assertRaisesRegex(ValueError, "steps"):
            PairedExperimentConfig(steps=0)
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            PairedExperimentConfig(sequence_length=1)


if __name__ == "__main__":
    unittest.main()
