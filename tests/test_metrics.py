"""Tests for metrics collection utilities."""

import pytest


class TestMetricBucket:
    """Test MetricBucket sliding window collector."""

    def test_empty_bucket_returns_zero(self):
        from tepid_h1.metrics import MetricBucket

        bucket = MetricBucket(window_size=10)
        assert bucket.count == 0
        assert bucket.mean == 0.0
        assert bucket.min_value == 0.0
        assert bucket.max_value == 0.0

    def test_add_and_query_values(self):
        from tepid_h1.metrics import MetricBucket

        bucket = MetricBucket(window_size=5)
        bucket.add(1.0)
        bucket.add(2.0)
        bucket.add(3.0)

        assert bucket.count == 3
        assert bucket.mean == 2.0
        assert bucket.min_value == 1.0
        assert bucket.max_value == 3.0

    def test_window_size_truncates_old_values(self):
        from tepid_h1.metrics import MetricBucket

        bucket = MetricBucket(window_size=3)
        bucket.add(1.0)
        bucket.add(2.0)
        bucket.add(3.0)
        bucket.add(4.0)  # should evict 1.0

        assert bucket.count == 3
        assert bucket.mean == pytest.approx(3.0)  # (2+3+4)/3
        assert bucket.min_value == 2.0
        assert bucket.max_value == 4.0

    def test_to_dict(self):
        from tepid_h1.metrics import MetricBucket

        bucket = MetricBucket(window_size=10)
        bucket.add(10.0)
        bucket.add(20.0)
        bucket.add(30.0)

        d = bucket.to_dict()
        assert d["count"] == 3
        assert d["mean"] == pytest.approx(20.0)
        assert d["min"] == pytest.approx(10.0)
        assert d["max"] == pytest.approx(30.0)


class TestTrainingMetrics:
    """Test TrainingMetrics aggregation."""

    def test_record_step_adds_all_metrics(self):
        from tepid_h1.metrics import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.record_step(loss=0.5, gradient_norm=1.2, learning_rate=1e-3, throughput=100.0)

        assert metrics.loss.count == 1
        assert metrics.gradient_norm.count == 1
        assert metrics.learning_rate.count == 1
        assert metrics.throughput.count == 1

    def test_summary_returns_dict(self):
        from tepid_h1.metrics import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.record_step(loss=0.5, gradient_norm=1.2, learning_rate=1e-3)

        summary = metrics.summary()
        assert "loss" in summary
        assert "gradient_norm" in summary
        assert "learning_rate" in summary
        assert "throughput" not in summary or summary["throughput"] is None

    def test_throughput_optional(self):
        from tepid_h1.metrics import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.record_step(loss=0.5, gradient_norm=1.2, learning_rate=1e-3, throughput=None)

        summary = metrics.summary()
        assert summary["throughput"] is None
