"""Tests for data module improvements."""

from tepid_h1.data import (
    BenchmarkSample,
    DomainMetrics,
    TextRecord,
    character_ngrams,
    compare_corpora,
    corpus_digest,
    normalize_text,
)


class TestDataExports:
    """Verify new exports are accessible."""

    def test_domain_metrics_accessible(self):
        """DomainMetrics should be importable from tepid_h1.data."""
        metrics = DomainMetrics(
            samples=1,
            utf8_bytes=100,
            characters=50,
            tokens=20,
            bytes_per_token=5.0,
            characters_per_token=2.5,
        )
        assert metrics.samples == 1
        assert metrics.bytes_per_token == 5.0

    def test_character_ngrams_exported(self):
        """character_ngrams should be importable from tepid_h1.data."""
        assert character_ngrams("abc", 2) == frozenset(("ab", "bc"))

    def test_normalize_text_exported(self):
        """normalize_text should be importable from tepid_h1.data."""
        assert normalize_text("  Hello  World  ") == "hello world"

    def test_corpus_digest_accessible(self):
        """corpus_digest should be importable from tepid_h1.data."""
        samples = (BenchmarkSample(domain="zh", text="test"),)
        digest = corpus_digest(samples)
        assert isinstance(digest, str)
        assert len(digest) == 64  # SHA-256 hex digest


class TestDataValidation:
    """Test data validation improvements."""

    def test_compare_corpora_with_normalized_unicode(self):
        """Unicode normalization should catch contamination."""
        training = (TextRecord("train-1", "Ｈｅｌｌｏ　Ｗｏｒｌｄ"),)
        benchmark = (TextRecord("bench-1", "hello world"),)
        report = compare_corpora(training, benchmark)
        assert not report.clean
        assert report.exact_matches == 1

    def test_clean_corpora_still_pass(self):
        """Clean corpora should still pass validation."""
        training = (TextRecord("train-1", "完全不同的内容 completely different"),)
        benchmark = (TextRecord("bench-1", "另一个完全不同的内容"),)
        report = compare_corpora(training, benchmark)
        assert report.clean
        assert report.contamination_rate == 0.0