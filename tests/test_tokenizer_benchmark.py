import unittest

from tepid_h1.data import BenchmarkSample, benchmark_candidate, select_candidate
from tepid_h1.data.tokenizer_benchmark import corpus_digest

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


if __name__ == "__main__":
    unittest.main()
