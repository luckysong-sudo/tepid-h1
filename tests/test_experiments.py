import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

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

        trial = report["trials"][0]
        self.assertEqual(trial["hybrid"]["trained_tokens"], 5)
        self.assertEqual(trial["baseline"]["trained_tokens"], 5)
        self.assertEqual(report["data"]["tokens_per_model_per_trial"], 5)
        self.assertTrue(report["parameters"]["hybrid"]["estimate_matches_actual"])
        self.assertTrue(report["parameters"]["baseline"]["estimate_matches_actual"])
        self.assertGreater(trial["hybrid"]["tokens_per_second"], 0)
        self.assertGreater(trial["baseline"]["tokens_per_second"], 0)
        self.assertEqual(report["environment"]["device_type"], "cpu")
        self.assertEqual(report["environment"]["dtype"], "float32")
        self.assertIsNone(trial["hybrid"]["peak_memory_bytes"])

    def test_same_seed_reproduces_data_and_loss(self):
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        config = PairedExperimentConfig(steps=1, sequence_length=5, seed=47)
        first = run_paired_smoke(config)
        second = run_paired_smoke(config)

        self.assertEqual(first["data"]["batch_sha256"], second["data"]["batch_sha256"])
        self.assertEqual(
            first["trials"][0]["hybrid"]["initial_loss"],
            second["trials"][0]["hybrid"]["initial_loss"],
        )
        self.assertEqual(
            first["trials"][0]["baseline"]["initial_loss"],
            second["trials"][0]["baseline"]["initial_loss"],
        )

    def test_invalid_experiment_config_is_rejected(self):
        from tepid_h1.experiments import PairedExperimentConfig

        with self.assertRaisesRegex(ValueError, "steps"):
            PairedExperimentConfig(steps=0)
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            PairedExperimentConfig(sequence_length=1)
        with self.assertRaisesRegex(ValueError, "trials"):
            PairedExperimentConfig(trials=0)
        with self.assertRaisesRegex(ValueError, "device"):
            PairedExperimentConfig(device="metal")
        with self.assertRaisesRegex(ValueError, "CPU"):
            PairedExperimentConfig(device="cpu", dtype="float16")

    def test_unavailable_cuda_fails_closed(self):
        from tepid_h1.experiments import PairedExperimentConfig, run_paired_smoke

        if torch.cuda.is_available():
            self.skipTest("CUDA is available in this test environment")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            run_paired_smoke(
                PairedExperimentConfig(steps=1, sequence_length=4, device="cuda")
            )

    def test_governed_corpus_binds_inventory_and_reports_uncertainty(self):
        from tepid_h1.experiments import (
            PairedExperimentConfig,
            load_governed_corpus,
            run_paired_smoke,
        )

        with tempfile.TemporaryDirectory() as directory:
            corpus_path, inventory_path = _write_governed_fixture(Path(directory))
            config = PairedExperimentConfig(
                steps=1,
                trials=2,
                batch_size=1,
                sequence_length=5,
                seed=53,
            )
            corpus = load_governed_corpus(
                corpus_path,
                inventory_path,
                config,
                vocab_size=128,
            )
            report = run_paired_smoke(config, corpus=corpus)

        self.assertEqual(report["experiment"], "paired_governed_corpus_smoke")
        self.assertEqual(report["data"]["inventory_id"], "test-inventory")
        self.assertEqual(report["data"]["source_id"], "test-source")
        self.assertEqual(len(report["data"]["inventory_file_sha256"]), 64)
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(report["aggregates"]["hybrid"]["loss_change"]["samples"], 2)
        throughput_ratio = report["aggregates"]["paired"][
            "baseline_over_hybrid_tokens_per_second"
        ]
        self.assertEqual(throughput_ratio["samples"], 2)
        self.assertGreater(throughput_ratio["ci95_low"], 0)

    def test_governed_corpus_rejects_checksum_mismatch(self):
        from tepid_h1.experiments import PairedExperimentConfig, load_governed_corpus

        with tempfile.TemporaryDirectory() as directory:
            corpus_path, inventory_path = _write_governed_fixture(
                Path(directory),
                checksum="0" * 64,
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_governed_corpus(
                    corpus_path,
                    inventory_path,
                    PairedExperimentConfig(steps=1, sequence_length=5),
                    vocab_size=128,
                )

    def test_governed_corpus_advances_from_resume_step(self):
        from tepid_h1.experiments import PairedExperimentConfig, load_governed_corpus

        with tempfile.TemporaryDirectory() as directory:
            corpus_path, inventory_path = _write_governed_fixture(Path(directory))
            config = PairedExperimentConfig(steps=1, sequence_length=5)
            first = load_governed_corpus(
                corpus_path,
                inventory_path,
                config,
                vocab_size=128,
            )
            resumed = load_governed_corpus(
                corpus_path,
                inventory_path,
                config,
                vocab_size=128,
                start_step=1,
            )

        self.assertEqual(first.start_step, 0)
        self.assertEqual(resumed.start_step, 1)
        self.assertNotEqual(first.batch_sha256, resumed.batch_sha256)
        self.assertEqual(first.batches[0].tolist(), [[1, 2, 3, 4, 5]])
        self.assertEqual(resumed.batches[0].tolist(), [[7, 8, 9, 10, 11]])

    def test_governed_training_and_validation_splits_are_isolated(self):
        from tepid_h1.experiments import (
            GovernedCorpus,
            validate_governed_split_isolation,
        )

        shared = {
            "batches": (torch.tensor([[1, 2, 3, 4]]),),
            "batch_sha256": "c" * 64,
            "start_step": 0,
            "inventory_file_sha256": "d" * 64,
            "inventory_id": "test-inventory",
            "records": 1,
            "domains": ("synthetic",),
        }
        training = GovernedCorpus(
            **shared,
            record_ids=("train-1",),
            file_sha256="a" * 64,
            source_id="training-source",
        )
        validation = GovernedCorpus(
            **shared,
            record_ids=("validation-1",),
            file_sha256="b" * 64,
            source_id="validation-source",
        )

        validate_governed_split_isolation(training, validation)
        with self.assertRaisesRegex(ValueError, "files"):
            validate_governed_split_isolation(
                training,
                replace(validation, file_sha256=training.file_sha256),
            )
        with self.assertRaisesRegex(ValueError, "source_id"):
            validate_governed_split_isolation(
                training,
                replace(validation, source_id=training.source_id),
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_governed_split_isolation(
                training,
                replace(validation, record_ids=training.record_ids),
            )


def _write_governed_fixture(
    directory: Path,
    *,
    checksum: str | None = None,
) -> tuple[Path, Path]:
    corpus_path = directory / "corpus.jsonl"
    corpus_text = (
        '{"id":"sample-1","source_id":"test-source","domain":"en",'
        '"token_ids":[1,2,3,4,5,6]}\n'
        '{"id":"sample-2","source_id":"test-source","domain":"code",'
        '"token_ids":[7,8,9,10,11,12]}\n'
    )
    corpus_path.write_text(corpus_text, encoding="utf-8")
    actual_checksum = hashlib.sha256(corpus_text.encode()).hexdigest()
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
