import tempfile
import unittest
from pathlib import Path

from tepid_h1.data import (
    CorpusStats,
    SplitIsolationReport,
    check_paired_corpus_isolation,
    load_paired_corpus_records,
    summarize_paired_corpus,
)


class CorpusStatsTests(unittest.TestCase):
    def _write_corpus(self, directory: Path, filename: str, lines: list[str]) -> Path:
        path = directory / filename
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        return path

    def test_summarize_example_paired_corpus(self):
        stats = summarize_paired_corpus("configs/paired_corpus.example.jsonl")
        self.assertIsInstance(stats, CorpusStats)
        self.assertEqual(stats.record_count, 6)
        self.assertEqual(stats.source_ids, ("synthetic-fixture-v1",))
        self.assertEqual(stats.domains, ("code", "en", "zh"))
        self.assertEqual(stats.records_by_source, {"synthetic-fixture-v1": 6})
        self.assertEqual(stats.records_by_domain, {"code": 2, "en": 2, "zh": 2})
        self.assertEqual(stats.total_token_ids, 72)
        self.assertEqual(stats.min_sequence_length, 12)
        self.assertEqual(stats.max_sequence_length, 12)
        self.assertEqual(stats.mean_sequence_length, 12.0)
        self.assertGreater(stats.unique_token_ids, 0)
        self.assertEqual(stats.token_id_min, 1)
        self.assertEqual(stats.token_id_max, 100)
        self.assertEqual(stats.duplicate_record_ids, ())

    def test_summarize_detects_duplicate_record_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_corpus(
                Path(directory),
                "duplicates.jsonl",
                [
                    '{"id":"r1","source_id":"s1","domain":"en","token_ids":[1,2,3]}',
                    '{"id":"r1","source_id":"s1","domain":"en","token_ids":[4,5,6]}',
                ],
            )
            stats = summarize_paired_corpus(path)
            self.assertEqual(stats.duplicate_record_ids, ("r1",))

    def test_summarize_handles_multiple_sources_and_domains(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_corpus(
                Path(directory),
                "sources.jsonl",
                [
                    '{"id":"r1","source_id":"s1","domain":"zh","token_ids":[1,2]}',
                    '{"id":"r2","source_id":"s1","domain":"en","token_ids":[3,4,5]}',
                    '{"id":"r3","source_id":"s2","domain":"code","token_ids":[6]}',
                ],
            )
            stats = summarize_paired_corpus(path)
            self.assertEqual(stats.record_count, 3)
            self.assertEqual(stats.source_ids, ("s1", "s2"))
            self.assertEqual(stats.domains, ("code", "en", "zh"))
            self.assertEqual(stats.records_by_source, {"s1": 2, "s2": 1})
            self.assertEqual(stats.records_by_domain, {"code": 1, "en": 1, "zh": 1})
            self.assertEqual(stats.total_token_ids, 6)
            self.assertEqual(stats.min_sequence_length, 1)
            self.assertEqual(stats.max_sequence_length, 3)
            self.assertAlmostEqual(stats.mean_sequence_length, 2.0)

    def test_summarize_rejects_empty_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_corpus(Path(directory), "empty.jsonl", [])
            with self.assertRaisesRegex(ValueError, "at least one record"):
                summarize_paired_corpus(path)

    def test_summarize_rejects_missing_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_corpus(
                Path(directory),
                "missing_field.jsonl",
                ['{"id":"r1","source_id":"s1","domain":"en"}'],
            )
            with self.assertRaisesRegex(ValueError, "token_ids"):
                summarize_paired_corpus(path)

    def test_summarize_rejects_non_integer_token_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_corpus(
                Path(directory),
                "invalid_token.jsonl",
                ['{"id":"r1","source_id":"s1","domain":"en","token_ids":[1,"two",3]}'],
            )
            with self.assertRaisesRegex(TypeError, "non-integer"):
                summarize_paired_corpus(path)

    def test_load_paired_corpus_records_returns_raw_dicts(self):
        records = load_paired_corpus_records("configs/paired_corpus.example.jsonl")
        self.assertEqual(len(records), 6)
        self.assertIn("id", records[0])
        self.assertIn("token_ids", records[0])

    def test_to_dict_is_serializable(self):
        import json

        stats = summarize_paired_corpus("configs/paired_corpus.example.jsonl")
        payload = json.dumps(stats.to_dict(), ensure_ascii=False)
        self.assertIn("record_count", payload)

    def test_check_paired_corpus_isolation_passes_on_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            training = self._write_corpus(
                Path(directory),
                "training.jsonl",
                ['{"id":"t1","source_id":"s1","domain":"en","token_ids":[1]}'],
            )
            validation = self._write_corpus(
                Path(directory),
                "validation.jsonl",
                ['{"id":"v1","source_id":"s2","domain":"zh","token_ids":[2]}'],
            )
            report = check_paired_corpus_isolation(training, validation)
            self.assertIsInstance(report, SplitIsolationReport)
            self.assertTrue(report.clean)
            self.assertEqual(report.shared_source_ids, ())
            self.assertEqual(report.shared_record_ids, ())

    def test_check_paired_corpus_isolation_detects_shared_source(self):
        with tempfile.TemporaryDirectory() as directory:
            training = self._write_corpus(
                Path(directory),
                "training.jsonl",
                ['{"id":"t1","source_id":"s1","domain":"en","token_ids":[1]}'],
            )
            validation = self._write_corpus(
                Path(directory),
                "validation.jsonl",
                ['{"id":"v1","source_id":"s1","domain":"zh","token_ids":[2]}'],
            )
            report = check_paired_corpus_isolation(training, validation)
            self.assertFalse(report.clean)
            self.assertEqual(report.shared_source_ids, ("s1",))

    def test_check_paired_corpus_isolation_detects_shared_record_id(self):
        with tempfile.TemporaryDirectory() as directory:
            training = self._write_corpus(
                Path(directory),
                "training.jsonl",
                ['{"id":"r1","source_id":"s1","domain":"en","token_ids":[1]}'],
            )
            validation = self._write_corpus(
                Path(directory),
                "validation.jsonl",
                ['{"id":"r1","source_id":"s2","domain":"zh","token_ids":[2]}'],
            )
            report = check_paired_corpus_isolation(training, validation)
            self.assertFalse(report.clean)
            self.assertEqual(report.shared_record_ids, ("r1",))

    def test_check_paired_corpus_isolation_rejects_same_file(self):
        path = "configs/paired_corpus.example.jsonl"
        with self.assertRaisesRegex(ValueError, "must be different"):
            check_paired_corpus_isolation(path, path)

    def test_split_isolation_report_to_dict(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            training = self._write_corpus(
                Path(directory),
                "training.jsonl",
                ['{"id":"t1","source_id":"s1","domain":"en","token_ids":[1]}'],
            )
            validation = self._write_corpus(
                Path(directory),
                "validation.jsonl",
                ['{"id":"v1","source_id":"s2","domain":"zh","token_ids":[2]}'],
            )
            report = check_paired_corpus_isolation(training, validation)
            payload = json.dumps(report.to_dict(), ensure_ascii=False)
            self.assertIn("clean", payload)
            self.assertIn("training_file_sha256", payload)


if __name__ == "__main__":
    unittest.main()
