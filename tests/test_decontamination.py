import unittest

from tepid_h1.data import TextRecord, compare_corpora
from tepid_h1.data.decontamination import character_ngrams, normalize_text


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


if __name__ == "__main__":
    unittest.main()
