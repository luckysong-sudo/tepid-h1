import unittest

from tepid_h1.data import TextRecord, compare_corpora
from tepid_h1.data.decontamination import (
    character_ngrams,
    file_sha256,
    load_text_records,
    normalize_text,
    text_sha256,
)


class DecontaminationTests(unittest.TestCase):
    def test_normalization_catches_unicode_and_whitespace_exact_match(self):
        training = (TextRecord("train-1", "Ｔｅｐｉｄ   H1\nAgent"),)
        benchmark = (TextRecord("bench-1", "tepid h1 agent"),)

        report = compare_corpora(training, benchmark)

        self.assertFalse(report.clean)
        self.assertEqual(report.exact_matches, 1)
        self.assertEqual(report.matches[0].match_type, "exact")
        self.assertNotIn("tepid h1 agent", report.to_dict()["matches"][0].values())

    def test_near_duplicate_is_detected(self):
        training = (
            TextRecord(
                "train-1",
                "The quick brown fox jumps over the lazy dog in the garden.",
            ),
        )
        benchmark = (
            TextRecord(
                "bench-1",
                "The quick brown fox jumps over a lazy dog in the garden.",
            ),
        )

        report = compare_corpora(training, benchmark, ngram_size=3, similarity_threshold=0.75)

        self.assertFalse(report.clean)
        self.assertEqual(report.near_matches, 1)
        self.assertGreaterEqual(report.matches[0].similarity, 0.75)

    def test_clean_corpora_pass(self):
        training = (TextRecord("train-1", "端侧模型需要低延迟推理。"),)
        benchmark = (TextRecord("bench-1", "数据库事务保证原子性。"),)

        report = compare_corpora(training, benchmark, ngram_size=3)

        self.assertTrue(report.clean)
        self.assertEqual(report.contamination_rate, 0.0)
        self.assertEqual(report.matches, ())

    def test_invalid_parameters_are_rejected(self):
        records = (TextRecord("one", "content"),)
        with self.assertRaisesRegex(ValueError, "ngram_size"):
            compare_corpora(records, records, ngram_size=0)
        with self.assertRaisesRegex(ValueError, "similarity_threshold"):
            compare_corpora(records, records, similarity_threshold=0.0)
        self.assertEqual(normalize_text(" A\n B "), "a b")
        self.assertEqual(character_ngrams("abc", 5), frozenset(("abc",)))

    def test_empty_training_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            compare_corpora((), (TextRecord("b", "content"),))

    def test_empty_benchmark_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            compare_corpora((TextRecord("t", "content"),), ())

    def test_character_ngrams_invalid_size(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            character_ngrams("test", 0)
        with self.assertRaisesRegex(ValueError, "positive"):
            character_ngrams("test", -1)

    def test_character_ngrams_single_char(self):
        ngrams = character_ngrams("abc", 1)
        self.assertEqual(ngrams, frozenset({"a", "b", "c"}))

    def test_character_ngrams_longer_than_text(self):
        ngrams = character_ngrams("ab", 5)
        self.assertEqual(ngrams, frozenset(("ab",)))

    def test_text_sha256_produces_hex_digest(self):
        digest = text_sha256("hello")
        self.assertEqual(len(digest), 64)
        int(digest, 16)  # Should not raise

    def test_normalize_text_casesfold_and_strip(self):
        self.assertEqual(normalize_text("  Hello  WORLD  "), "hello world")

    def test_load_text_records_requires_text_field(self):
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1"}\n')
            path = f.name
        try:
            with self.assertRaisesRegex(TypeError, "text"):
                load_text_records(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_text_records_requires_non_empty_text(self):
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "text": "   "}\n')
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "empty text"):
                load_text_records(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_text_records_requires_valid_id(self):
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "", "text": "hello"}\n')
            path = f.name
        try:
            with self.assertRaisesRegex(TypeError, "id"):
                load_text_records(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_text_records_rejects_duplicate_ids(self):
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "text": "hello"}\n')
            f.write('{"id": "1", "text": "world"}\n')
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_text_records(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_text_records_requires_at_least_one_record(self):
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('')
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "at least one"):
                load_text_records(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_text_records_rejects_invalid_json(self):
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('not json\n')
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_text_records(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_file_sha256_produces_hex_digest(self):
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('hello world')
            path = f.name
        try:
            digest = file_sha256(path)
            self.assertEqual(len(digest), 64)
            int(digest, 16)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
