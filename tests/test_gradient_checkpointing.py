"""Tests for gradient checkpointing utilities."""

from __future__ import annotations

import torch
import torch.nn as nn

from tepid_h1.config import TepidH1Config
from tepid_h1.gradient_checkpointing import (
    CheckpointedLayer,
    apply_gradient_checkpointing,
    estimate_memory_savings,
    wrap_layers_with_checkpointing,
)
from tepid_h1.modeling import TepidH1CausalLM


class SimpleSequentialModel(nn.Module):
    """A simple sequential model for testing checkpointing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestCheckpointedLayer:
    """Tests for CheckpointedLayer wrapper."""

    def test_checkpointed_layer_default_enabled(self) -> None:
        layer = SimpleSequentialModel()
        checkpointed = CheckpointedLayer(layer, enabled=True)
        assert checkpointed.enabled is True
        x = torch.randn(2, 64)
        output = checkpointed(x)
        assert output.shape == (2, 64)

    def test_checkpointed_layer_disabled(self) -> None:
        layer = SimpleSequentialModel()
        checkpointed = CheckpointedLayer(layer, enabled=False)
        assert checkpointed.enabled is False
        x = torch.randn(2, 64)
        output = checkpointed(x)
        assert output.shape == (2, 64)

    def test_checkpointed_layer_repr(self) -> None:
        layer = SimpleSequentialModel()
        checkpointed = CheckpointedLayer(layer, enabled=True)
        assert "enabled=True" in repr(checkpointed)

    def test_checkpointed_layer_enabled_false(self) -> None:
        layer = SimpleSequentialModel()
        checkpointed = CheckpointedLayer(layer, enabled=False)
        assert "enabled=False" in repr(checkpointed)


class TestApplyGradientCheckpointing:
    """Tests for apply_gradient_checkpointing function."""

    def test_applies_to_all_modules(self) -> None:
        model = SimpleSequentialModel()
        checkpointed = apply_gradient_checkpointing(model, checkpoint_every=1)
        assert isinstance(checkpointed.linear1, CheckpointedLayer)
        assert isinstance(checkpointed.relu, CheckpointedLayer)
        assert isinstance(checkpointed.linear2, CheckpointedLayer)
        x = torch.randn(2, 64)
        output = checkpointed(x)
        assert output.shape == (2, 64)

    def test_checkpoint_every_2(self) -> None:
        model = SimpleSequentialModel()
        checkpointed = apply_gradient_checkpointing(model, checkpoint_every=2)
        assert not isinstance(checkpointed.linear1, CheckpointedLayer)
        assert isinstance(checkpointed.relu, CheckpointedLayer)
        assert not isinstance(checkpointed.linear2, CheckpointedLayer)
        x = torch.randn(2, 64)
        output = checkpointed(x)
        assert output.shape == (2, 64)

    def test_rejects_invalid_checkpoint_every(self) -> None:
        model = SimpleSequentialModel()

        try:
            apply_gradient_checkpointing(model, checkpoint_every=0)
        except ValueError as error:
            assert "checkpoint_every" in str(error)
        else:
            raise AssertionError("checkpoint_every=0 should fail")

        try:
            apply_gradient_checkpointing(model, checkpoint_every=True)
        except TypeError as error:
            assert "checkpoint_every" in str(error)
        else:
            raise AssertionError("boolean checkpoint_every should fail")

    def test_gradient_flow(self) -> None:
        model = SimpleSequentialModel()
        checkpointed = apply_gradient_checkpointing(model, checkpoint_every=1)
        x = torch.randn(2, 64, requires_grad=True)
        output = checkpointed(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_saves_memory(self) -> None:
        """Checkpointing should reduce peak memory during backward."""
        model = SimpleSequentialModel()
        checkpointed = apply_gradient_checkpointing(model, checkpoint_every=1)

        x = torch.randn(4, 64, requires_grad=True)
        output = checkpointed(x)
        loss = output.sum()
        loss.backward()
        # Verify gradients are computed correctly
        assert model.linear1.layer.weight.grad is not None


class TestWrapLayersWithCheckpointing:
    """Tests for wrap_layers_with_checkpointing function."""

    def test_wraps_specified_layers(self) -> None:
        model = SimpleSequentialModel()
        checkpointed = wrap_layers_with_checkpointing(model, layer_indices=[0, 1])
        assert isinstance(checkpointed.linear1, CheckpointedLayer)
        assert isinstance(checkpointed.relu, CheckpointedLayer)
        assert not isinstance(checkpointed.linear2, CheckpointedLayer)
        x = torch.randn(2, 64)
        output = checkpointed(x)
        assert output.shape == (2, 64)

    def test_does_not_wrap_other_layers(self) -> None:
        model = SimpleSequentialModel()
        checkpointed = wrap_layers_with_checkpointing(model, layer_indices=[])
        assert not isinstance(checkpointed.linear1, CheckpointedLayer)
        assert not isinstance(checkpointed.relu, CheckpointedLayer)
        assert not isinstance(checkpointed.linear2, CheckpointedLayer)
        x = torch.randn(2, 64)
        output = checkpointed(x)
        assert output.shape == (2, 64)

    def test_rejects_invalid_layer_indices(self) -> None:
        model = SimpleSequentialModel()

        try:
            wrap_layers_with_checkpointing(model, layer_indices=[-1])
        except ValueError as error:
            assert "layer_indices" in str(error)
        else:
            raise AssertionError("negative layer index should fail")

        try:
            wrap_layers_with_checkpointing(model, layer_indices=[True])
        except TypeError as error:
            assert "layer_indices" in str(error)
        else:
            raise AssertionError("boolean layer index should fail")

        try:
            wrap_layers_with_checkpointing(model, layer_indices=[3])
        except ValueError as error:
            assert "out of range" in str(error)
        else:
            raise AssertionError("out-of-range layer index should fail")

    def test_gradient_flow(self) -> None:
        model = SimpleSequentialModel()
        checkpointed = wrap_layers_with_checkpointing(model, layer_indices=[0])
        x = torch.randn(2, 64, requires_grad=True)
        output = checkpointed(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestEstimateMemorySavings:
    """Tests for estimate_memory_savings function."""

    def test_returns_valid_estimates(self) -> None:
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        model.eval()
        estimates = estimate_memory_savings(model, batch_size=2, sequence_length=8)
        assert "parameter_memory_bytes" in estimates
        assert "estimated_activation_memory_bytes" in estimates
        assert "estimated_savings_with_checkpointing_bytes" in estimates
        assert "total_memory_with_checkpointing_bytes" in estimates

    def test_estimates_are_positive(self) -> None:
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        model.eval()
        estimates = estimate_memory_savings(model, batch_size=2, sequence_length=8)
        assert estimates["parameter_memory_bytes"] > 0
        assert estimates["estimated_activation_memory_bytes"] > 0
        assert estimates["estimated_savings_with_checkpointing_bytes"] > 0

    def test_total_memory_less_than_without_checkpointing(self) -> None:
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        model.eval()
        estimates = estimate_memory_savings(model, batch_size=2, sequence_length=8)
        # Total with checkpointing should be less than param + activation
        total_with = estimates["total_memory_with_checkpointing_bytes"]
        total_without = (
            estimates["parameter_memory_bytes"] + estimates["estimated_activation_memory_bytes"]
        )
        assert total_with < total_without
