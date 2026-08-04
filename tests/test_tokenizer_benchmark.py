import unittest

from tepid_h1.data import BenchmarkSample, benchmark_candidate, select_candidate
from tepid_h1.data.tokenizer_benchmark import cached_tokenize, corpus_digest

SAMPLES = (
    BenchmarkSample(domain="zh", text="端侧智能"),
    BenchmarkSample(domain="en", text="edge intelligence"),
    BenchmarkSample(domain="code", text="def run(): return True"),
)


class TokenizerBenchmarkTests(unittest.TestCase):
    def test_benchmark_records_domain_metrics_and_digest(self):
        ticks = iter((10.0, 12.0))
        result = benchmark_candidate(
            name="fixture-64k",
            vocab_size=64_000,
            encode=lambda text: list(range(max(1, len(text) // 2))),
            samples=SAMPLES,
            clock=lambda: next(ticks),
        )

        self.assertEqual(set(result["domains"]), {"zh", "en", "code"})
        self.assertGreater(result["tokens_per_second"], 0)
        self.assertGreater(result["utf8_bytes_per_second"], 0)
        self.assertEqual(len(corpus_digest(SAMPLES)), 64)

    def test_selection_requires_all_three_vocab_sizes(self):
        candidate = {
            "name": "only",
            "vocab_size": 64_000,
            "tokens_per_second": 100,
            "domains": {},
        }
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_candidate([candidate])

    def test_selection_balances_compression_and_throughput(self):
        candidates = []
        for name, vocab_size, compression, throughput in (
            ("64k", 64_000, 2.0, 120.0),
            ("80k", 80_000, 3.0, 110.0),
            ("96k", 96_000, 3.1, 60.0),
        ):
            candidates.append(
                {
                    "name": name,
                    "vocab_size": vocab_size,
                    "tokens_per_second": throughput,
                    "utf8_bytes_per_second": throughput,
                    "domains": {
                        domain: {"bytes_per_token": compression} for domain in ("zh", "en", "code")
                    },
                }
            )

        selection = select_candidate(candidates)
        self.assertEqual(selection["selected"], "80k")
        self.assertEqual(selection["selected_vocab_size"], 80_000)

    def test_cached_tokenize_reuses_result_for_matching_content_and_name(self):
        calls = 0

        def encode(text):
            nonlocal calls
            calls += 1
            return [len(text)]

        first, _ = cached_tokenize(encode, "tokenizer fixture", name="fixture")
        second, _ = cached_tokenize(encode, "tokenizer fixture", name="fixture")

        self.assertEqual(first, (17,))
        self.assertEqual(second, (17,))
        self.assertEqual(calls, 1)

    def test_load_corpus_requires_text_field(self):
        import tempfile
        from pathlib import Path
        from tepid_h1.data.tokenizer_benchmark import load_corpus

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "1", "domain": "en"}\n')
            path = f.name
        try:
            with self.assertRaisesRegex(TypeError, "text"):
                load_corpus(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_corpus_requires_valid_domain(self):
        import tempfile
        from pathlib import Path
        from tepid_h1.data.tokenizer_benchmark import load_corpus

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "hello", "domain": "xx"}\n')
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "domain"):
                load_corpus(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_corpus_requires_non_empty_text(self):
        import tempfile
        from pathlib import Path
        from tepid_h1.data.tokenizer_benchmark import load_corpus

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "", "domain": "en"}\n')
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "empty"):
                load_corpus(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_corpus_requires_at_least_one_sample(self):
        import tempfile
        from pathlib import Path
        from tepid_h1.data.tokenizer_benchmark import load_corpus

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('')
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "at least one"):
                load_corpus(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_corpus_requires_all_domains(self):
        import tempfile
        from pathlib import Path
        from tepid_h1.data.tokenizer_benchmark import load_corpus

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "hello", "domain": "en"}\n')
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "missing domains"):
                load_corpus(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_corpus_rejects_invalid_json(self):
        import tempfile
        from pathlib import Path
        from tepid_h1.data.tokenizer_benchmark import load_corpus

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('not json\n')
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_corpus(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_benchmark_candidate_requires_valid_vocab_size(self):
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            benchmark_candidate(
                name="test",
                vocab_size=50_000,
                encode=lambda text: [1, 2, 3],
                samples=SAMPLES,
            )

    def test_benchmark_candidate_rejects_empty_samples(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            benchmark_candidate(
                name="test",
                vocab_size=64_000,
                encode=lambda text: [1, 2, 3],
                samples=[],
            )

    def test_select_candidate_rejects_missing_domains(self):
        candidates = []
        for vocab_size in (64_000, 80_000, 96_000):
            candidates.append({
                "name": f"{vocab_size}k",
                "vocab_size": vocab_size,
                "tokens_per_second": 100,
                "utf8_bytes_per_second": 100,
                "domains": {"zh": {"bytes_per_token": 2.0}, "en": {"bytes_per_token": 2.0}},
            })
        with self.assertRaisesRegex(ValueError, "missing domains"):
            select_candidate(candidates)


if __name__ == "__main__":
    unittest.main()
