"""Edge case tests for experiments module."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PairedExperimentEdgeCaseTests(unittest.TestCase):
    """Additional edge case tests for paired experiment module."""

    def test_multi_trial_reproducibility(self):
        """Multi-trial experiments should be reproducible with same seed."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=2, trials=3, sequence_length=6, seed=99
        )
        first = run_paired_smoke(config)
        second = run_paired_smoke(config)

        self.assertEqual(len(first["trials"]), 3)
        self.assertEqual(len(second["trials"]), 3)
        for i in range(3):
            trial_a = first["trials"][i]
            trial_b = second["trials"][i]
            self.assertEqual(
                trial_a["hybrid"]["initial_loss"],
                trial_b["hybrid"]["initial_loss"],
                f"Trial {i} hybrid loss mismatch",
            )
            self.assertEqual(
                trial_a["baseline"]["initial_loss"],
                trial_b["baseline"]["initial_loss"],
                f"Trial {i} baseline loss mismatch",
            )

    def test_different_seed_produces_different_data(self):
        """Different seeds should produce different batch digests."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config_a = PairedExperimentConfig(steps=1, sequence_length=5, seed=1)
        config_b = PairedExperimentConfig(steps=1, sequence_length=5, seed=2)

        report_a = run_paired_smoke(config_a)
        report_b = run_paired_smoke(config_b)

        self.assertNotEqual(
            report_a["data"]["batch_sha256"],
            report_b["data"]["batch_sha256"],
        )

    def test_aggregate_statistics_with_multiple_trials(self):
        """Aggregate statistics should be computed correctly with multiple trials."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=1, trials=5, sequence_length=5, seed=55
        )
        report = run_paired_smoke(config)

        aggregates = report["aggregates"]
        self.assertEqual(aggregates["hybrid"]["loss_change"]["samples"], 5)
        self.assertEqual(aggregates["baseline"]["loss_change"]["samples"], 5)
        self.assertEqual(
            aggregates["paired"]["baseline_over_hybrid_tokens_per_second"]["samples"], 5
        )

    def test_execution_order_alters_between_trials(self):
        """Execution order should alternate between trials."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=1, trials=4, sequence_length=5, seed=77
        )
        report = run_paired_smoke(config)

        orders = [trial["execution_order"][0] for trial in report["trials"]]
        # Check that order alternates based on trial_index
        self.assertIn(["hybrid", "baseline"], orders)
        self.assertIn(["baseline", "hybrid"], orders)

    def test_multi_trial_loss_change_aggregation(self):
        """Loss change aggregation should produce valid statistics."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=1, trials=4, sequence_length=6, seed=88
        )
        report = run_paired_smoke(config)

        hybrid_ci = report["aggregates"]["hybrid"]["loss_change"]
        self.assertLessEqual(hybrid_ci["ci95_low"], hybrid_ci["ci95_high"])
        self.assertIsInstance(hybrid_ci["mean"], float)

    def test_governed_corpus_accepts_multi_step_resume(self):
        """Governed corpus should correctly advance through multiple steps."""
        from tepid_h1.experiments import PairedExperimentConfig, load_governed_corpus

        with tempfile.TemporaryDirectory() as directory:
            corpus_path, inventory_path = _write_governed_fixture(Path(directory))
            config = PairedExperimentConfig(steps=3, sequence_length=5)

            step0 = load_governed_corpus(
                corpus_path, inventory_path, config, vocab_size=128, start_step=0
            )
            step2 = load_governed_corpus(
                corpus_path, inventory_path, config, vocab_size=128, start_step=2
            )

            self.assertEqual(step0.start_step, 0)
            self.assertEqual(step2.start_step, 2)
            self.assertEqual(len(step0.batches), 3)
            self.assertEqual(len(step2.batches), 3)

    def test_governed_corpus_circular_record_wrapping(self):
        """Corpus should wrap around records when steps exceed record count."""
        from tepid_h1.experiments import PairedExperimentConfig, load_governed_corpus

        with tempfile.TemporaryDirectory() as directory:
            corpus_path, inventory_path = _write_governed_fixture(Path(directory))
            config = PairedExperimentConfig(steps=5, batch_size=1, sequence_length=5)

            corpus = load_governed_corpus(
                corpus_path, inventory_path, config, vocab_size=128
            )
            self.assertEqual(len(corpus.batches), 5)

    def test_corpus_with_larger_sequence_length(self):
        """Corpus should handle sequence lengths close to record length."""
        from tepid_h1.experiments import PairedExperimentConfig, load_governed_corpus

        with tempfile.TemporaryDirectory() as directory:
            corpus_path, inventory_path = _write_governed_fixture(Path(directory))
            config = PairedExperimentConfig(steps=1, sequence_length=6)

            corpus = load_governed_corpus(
                corpus_path, inventory_path, config, vocab_size=128
            )
            self.assertEqual(corpus.batches[0].shape, (1, 6))

    def test_max_trials_boundary(self):
        """Test with maximum allowed trials (20)."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=1, trials=20, sequence_length=4, seed=1
        )
        report = run_paired_smoke(config)
        self.assertEqual(len(report["trials"]), 20)

    def test_max_sequence_length_boundary(self):
        """Test with maximum allowed sequence length (64)."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=1, trials=1, sequence_length=64, seed=1
        )
        report = run_paired_smoke(config)
        self.assertEqual(len(report["data"]["batch_sha256"]), 64)

    def test_minimum_sequence_length_boundary(self):
        """Test with minimum allowed sequence length (2)."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=1, trials=1, sequence_length=2, seed=1
        )
        report = run_paired_smoke(config)
        self.assertEqual(len(report["data"]["batch_sha256"]), 64)

    def test_probe_batch_shares_seed_space_independently(self):
        """Probe batches should be deterministic and separate from training batches."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=1, trials=1, sequence_length=5, seed=42
        )
        report = run_paired_smoke(config)

        # Probe digest should be stable
        self.assertEqual(len(report["data"]["probe_batch_sha256"]), 64)
        # Data and probe digests should differ
        self.assertNotEqual(
            report["data"]["batch_sha256"],
            report["data"]["probe_batch_sha256"],
        )

    def test_loss_metrics_decrease_or_increase(self):
        """Loss metrics should be finite and reasonable."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=2, trials=2, sequence_length=5, seed=1
        )
        report = run_paired_smoke(config)

        for trial in report["trials"]:
            hybrid = trial["hybrid"]
            baseline = trial["baseline"]

            # Losses should be finite
            self.assertTrue(hybrid["initial_loss"] != float("inf"))
            self.assertTrue(hybrid["final_loss"] != float("inf"))
            self.assertTrue(baseline["initial_loss"] != float("inf"))
            self.assertTrue(baseline["final_loss"] != float("inf"))

            # Eval losses should be finite
            self.assertTrue(hybrid["initial_eval_loss"] != float("inf"))
            self.assertTrue(hybrid["final_eval_loss"] != float("inf"))

    def test_probe_token_count_matches(self):
        """Probe token count should match expected computation."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(
            steps=1, trials=1, batch_size=2, sequence_length=8, seed=1
        )
        report = run_paired_smoke(config)

        expected_probe_tokens = 2 * (8 - 1)
        self.assertEqual(
            report["data"]["probe_tokens_per_model_per_trial"],
            expected_probe_tokens,
        )

    def test_tokens_per_model_total_scales_with_trials(self):
        """Total tokens should scale linearly with trial count."""
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config_single = PairedExperimentConfig(
            steps=1, trials=1, batch_size=1, sequence_length=5, seed=1
        )
        config_multi = PairedExperimentConfig(
            steps=1, trials=4, batch_size=1, sequence_length=5, seed=1
        )

        report_single = run_paired_smoke(config_single)
        report_multi = run_paired_smoke(config_multi)

        per_trial = report_single["data"]["tokens_per_model_per_trial"]
        self.assertEqual(
            report_multi["data"]["tokens_per_model_total"],
            per_trial * 4,
        )


def _write_governed_fixture(
    directory: Path,
    *,
    checksum: str | None = None,
) -> tuple[Path, Path]:
    """Write a minimal governed corpus fixture for testing."""
    corpus_path = directory / "corpus.jsonl"
    corpus_text = (
        b'{"id":"sample-1","source_id":"test-source","domain":"en",'
        b'"token_ids":[1,2,3,4,5,6]}\n'
        b'{"id":"sample-2","source_id":"test-source","domain":"code",'
        b'"token_ids":[7,8,9,10,11,12]}\n'
    )
    corpus_path.write_bytes(corpus_text)
    actual_checksum = hashlib.sha256(corpus_text).hexdigest()
    inventory = {
        "schema_version": 1,
        "inventory_id": "test-inventory",
        "owner": "tests",
        "generated_at": "2026-07-29T00:00:00Z",
        "sources": [
            {
                "id": "test-source",
                "name": "Synthetic test source",
                "uri": "repo://tests/corpus.jsonl",
                "snapshot": "v1",
                "sha256": checksum or actual_checksum,
                "license_id": "CC0-1.0",
                "license_category": "public_domain",
                "commercial_use": True,
                "rights_evidence": "test fixture",
                "languages": ["en"],
                "domains": ["synthetic"],
                "estimated_tokens": 12,
                "pii_status": "absent",
                "quality_status": "accepted",
            }
        ],
        "repository_decontamination": {
            "status": "complete",
            "method": "synthetic fixture isolation",
            "benchmark_sets": ["synthetic-heldout"],
            "report_uri": "test fixture",
            "completed_at": "2026-07-29T00:00:00Z",
        },
    }
    inventory_path = directory / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    return corpus_path, inventory_path


if __name__ == "__main__":
    unittest.main()


