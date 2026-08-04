"""Tests for training callbacks and monitoring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from tepid_h1.callbacks import (
    EarlyStopper,
    LossTracker,
    TrainingCallback,
    TrainingMetricsBuffer,
    TrainingRunner,
)


class TestTrainingMetricsBuffer:
    def test_empty_summary(self) -> None:
        buf = TrainingMetricsBuffer()
        assert buf.summary() == {"samples": 0}

    def test_record_and_summary(self) -> None:
        buf = TrainingMetricsBuffer()
        buf.record(1.0, 0.5, 10.0)
        buf.record(0.8, 0.3, 12.0)
        s = buf.summary()
        assert s["samples"] == 2
        assert abs(s["loss_mean"] - 0.9) < 1e-6
        assert s["loss_min"] == 0.8
        assert s["loss_max"] == 1.0
        assert abs(s["grad_norm_mean"] - 0.4) < 1e-6
        assert abs(s["throughput_mean"] - 11.0) < 1e-6

    def test_window_size_limit(self) -> None:
        buf = TrainingMetricsBuffer()
        for i in range(150):
            buf.record(float(i), 0.1, 1.0)
        s = buf.summary()
        assert s["samples"] == 100

    def test_is_stable_false_when_few_samples(self) -> None:
        buf = TrainingMetricsBuffer()
        buf.record(1.0, 0.1, 1.0)
        assert buf.is_stable() is False

    def test_is_stable_true(self) -> None:
        buf = TrainingMetricsBuffer()
        for _ in range(15):
            buf.record(0.5, 0.1, 1.0)
        assert buf.is_stable(tolerance=0.01) is True

    def test_is_stable_false_with_variance(self) -> None:
        buf = TrainingMetricsBuffer()
        for i in range(15):
            buf.record(float(i) * 0.1, 0.1, 1.0)
        assert buf.is_stable(tolerance=0.01) is False

    def test_is_stable_custom_window(self) -> None:
        buf = TrainingMetricsBuffer()
        for i in range(20):
            buf.record(1.0 + float(i) * 0.0001, 0.1, 1.0)
        assert buf.is_stable(tolerance=0.01, window=5) is True


class TestEarlyStopper:
    def test_init_valid(self) -> None:
        stopper = EarlyStopper(patience=3, min_delta=0.01, mode="min")
        assert stopper.patience == 3
        assert stopper.min_delta == 0.01
        assert stopper.mode == "min"
        assert stopper.should_stop is False
        assert stopper.best_value is None

    def test_init_invalid_patience(self) -> None:
        with pytest.raises(ValueError, match="patience must be non-negative"):
            EarlyStopper(patience=-1)

    def test_init_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            EarlyStopper(mode="invalid")

    def test_min_mode_improvement(self) -> None:
        stopper = EarlyStopper(patience=2, mode="min")
        assert stopper.update(1.0) is True
        assert stopper.best_value == 1.0
        assert stopper.update(0.8) is True
        assert stopper.best_value == 0.8

    def test_min_mode_no_improvement(self) -> None:
        stopper = EarlyStopper(patience=2, mode="min", min_delta=0.5)
        assert stopper.update(1.0) is True
        assert stopper.update(1.2) is False
        assert stopper.update(1.5) is False
        assert stopper.should_stop is True

    def test_max_mode(self) -> None:
        stopper = EarlyStopper(patience=2, mode="max")
        assert stopper.update(0.5) is True
        assert stopper.update(0.6) is True
        assert stopper.update(0.4) is False
        assert stopper.update(0.3) is False
        assert stopper.should_stop is True

    def test_min_delta_threshold(self) -> None:
        stopper = EarlyStopper(patience=2, min_delta=0.1, mode="min")
        assert stopper.update(1.0) is True
        assert stopper.update(0.95) is False  # below min_delta
        assert stopper.update(0.89) is True  # above min_delta

    def test_counter_reset_on_improvement(self) -> None:
        stopper = EarlyStopper(patience=2, mode="min", min_delta=0.05)
        assert stopper.update(1.0) is True
        assert stopper.update(0.94) is True  # improved by 0.06
        assert stopper.update(0.95) is False
        assert stopper.update(0.96) is False
        assert stopper.should_stop is True


class TestLossTracker:
    def test_empty_report(self) -> None:
        tracker = LossTracker()
        assert tracker.report()["status"] == "no_data"
        assert tracker.losses == []
        assert tracker.steps == []
        assert tracker.elapsed_seconds >= 0

    def test_record_and_report(self) -> None:
        tracker = LossTracker()
        tracker.record(0, 2.0)
        tracker.record(1, 1.5)
        tracker.record(2, 1.0)
        r = tracker.report()
        assert r["total_steps"] == 3
        assert r["current_loss"] == 1.0
        assert r["initial_loss"] == 2.0
        assert abs(r["total_reduction"] - 1.0) < 1e-6
        assert abs(r["reduction_percent"] - 50.0) < 1e-6
        assert r["elapsed_seconds"] >= 0

    def test_convergence_rate(self) -> None:
        tracker = LossTracker()
        for i in range(15):
            tracker.record(i, 2.0 - i * 0.1)
        rate = tracker.convergence_rate(window=10)
        assert rate is not None
        assert rate > 0

    def test_convergence_rate_insufficient(self) -> None:
        tracker = LossTracker()
        tracker.record(0, 1.0)
        tracker.record(1, 0.9)
        assert tracker.convergence_rate(window=10) is None

    def test_losses_and_steps_lists(self) -> None:
        tracker = LossTracker()
        tracker.record(5, 3.0)
        tracker.record(10, 2.0)
        assert tracker.losses == [3.0, 2.0]
        assert tracker.steps == [5, 10]


class TestTrainingCallback:
    def test_no_callbacks(self) -> None:
        cb = TrainingCallback()
        assert cb.on_step is None
        assert cb.on_epoch is None
        assert cb.on_checkpoint is None
        assert cb.on_error is None


class TestTrainingRunner:
    def test_train_step(self) -> None:
        model = MagicMock()
        output = MagicMock()
        output.loss = torch.tensor(0.5, requires_grad=True)
        model.return_value = output
        optimizer = MagicMock()
        optimizer.step.return_value = None
        optimizer.zero_grad.return_value = None
        optimizer.param_groups = [{"params": [], "lr": 0.001}]
        runner = TrainingRunner(model, optimizer)
        input_ids = torch.randint(0, 10, (2, 4))
        metrics = runner.train_step(input_ids)
        assert "loss" in metrics
        assert metrics["step"] == 1

    def test_train_step_nan_loss_raises(self) -> None:
        model = MagicMock()
        output = MagicMock()
        output.loss = torch.tensor(float("nan"), requires_grad=True)
        model.return_value = output
        optimizer = MagicMock()
        optimizer.step.return_value = None
        optimizer.zero_grad.return_value = None
        optimizer.param_groups = [{"params": [], "lr": 0.001}]
        runner = TrainingRunner(model, optimizer)
        input_ids = torch.randint(0, 10, (2, 4))
        with pytest.raises(RuntimeError, match="NaN or Inf"):
            runner.train_step(input_ids)

    def test_train_step_no_loss_raises(self) -> None:
        model = MagicMock()
        output = MagicMock()
        output.loss = None
        model.return_value = output
        optimizer = MagicMock()
        runner = TrainingRunner(model, optimizer)
        input_ids = torch.randint(0, 10, (2, 4))
        with pytest.raises(RuntimeError, match="model did not return a loss"):
            runner.train_step(input_ids)

    def test_train_step_rejects_non_positive_gradient_norm_before_forward(self) -> None:
        model = MagicMock()
        optimizer = MagicMock()
        runner = TrainingRunner(model, optimizer)
        input_ids = torch.randint(0, 10, (2, 4))

        with pytest.raises(ValueError, match="max_gradient_norm"):
            runner.train_step(input_ids, max_gradient_norm=0)

        model.assert_not_called()
        optimizer.zero_grad.assert_not_called()
        optimizer.step.assert_not_called()

    def test_callbacks_invoked(self) -> None:
        step_calls: list[int] = []
        cb = TrainingCallback(
            on_step=lambda s, m: step_calls.append(s),
        )
        model = MagicMock()
        output = MagicMock()
        output.loss = torch.tensor(0.5, requires_grad=True)
        model.return_value = output
        optimizer = MagicMock()
        optimizer.step.return_value = None
        optimizer.zero_grad.return_value = None
        optimizer.param_groups = [{"params": [], "lr": 0.001}]
        runner = TrainingRunner(model, optimizer, callbacks=[cb])
        input_ids = torch.randint(0, 10, (2, 4))
        runner.train_step(input_ids)
        assert step_calls == [1]

    def test_checkpoint(self, tmp_path: Path) -> None:
        model = MagicMock()
        model.state_dict.return_value = {"weight": torch.tensor([1.0])}
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {"param_groups": []}
        runner = TrainingRunner(model, optimizer)
        path = tmp_path / "ckpt.pt"
        state = runner.checkpoint(path)
        assert path.exists()
        assert "model_state" in state

    def test_checkpoint_callback(self, tmp_path: Path) -> None:
        ckpt_calls: list[int] = []
        cb = TrainingCallback(on_checkpoint=lambda s, m: ckpt_calls.append(s))
        model = MagicMock()
        model.state_dict.return_value = {}
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {}
        runner = TrainingRunner(model, optimizer, callbacks=[cb])
        path = tmp_path / "ckpt.pt"
        runner.checkpoint(path)
        assert ckpt_calls == [0]

    def test_train_epoch(self) -> None:
        model = MagicMock()
        output = MagicMock()
        output.loss = torch.tensor(0.5, requires_grad=True)
        model.return_value = output
        optimizer = MagicMock()
        optimizer.step.return_value = None
        optimizer.zero_grad.return_value = None
        optimizer.param_groups = [{"params": [], "lr": 0.001}]
        runner = TrainingRunner(model, optimizer)
        batches = [torch.randint(0, 10, (2, 4))]
        metrics = runner.train_epoch(batches)
        assert "avg_loss" in metrics
        assert metrics["steps_per_second"] > 0

    def test_train_epoch_labels(self) -> None:
        model = MagicMock()
        output = MagicMock()
        output.loss = torch.tensor(0.5, requires_grad=True)
        model.return_value = output
        optimizer = MagicMock()
        optimizer.step.return_value = None
        optimizer.zero_grad.return_value = None
        optimizer.param_groups = [{"params": [], "lr": 0.001}]
        runner = TrainingRunner(model, optimizer)
        batches = [torch.randint(0, 10, (2, 4))]
        labels = [torch.randint(0, 10, (2, 4))]
        metrics = runner.train_epoch(batches, labels_batches=labels)
        assert "avg_loss" in metrics

    def test_train_epoch_mismatched_labels_raises(self) -> None:
        model = MagicMock()
        output = MagicMock()
        output.loss = torch.tensor(0.5)
        model.return_value = output
        optimizer = MagicMock()
        optimizer.param_groups = [{"lr": 0.001}]
        runner = TrainingRunner(model, optimizer)
        batches = [torch.randint(0, 10, (2, 4))]
        labels = [torch.randint(0, 10, (2, 4)), torch.randint(0, 10, (2, 4))]
        with pytest.raises(ValueError, match="labels_batches must match"):
            runner.train_epoch(batches, labels_batches=labels)

    def test_train_epoch_rejects_empty_batches_before_forward(self) -> None:
        model = MagicMock()
        optimizer = MagicMock()
        runner = TrainingRunner(model, optimizer)

        with pytest.raises(ValueError, match="batches"):
            runner.train_epoch([])

        model.assert_not_called()
        optimizer.zero_grad.assert_not_called()
        optimizer.step.assert_not_called()
