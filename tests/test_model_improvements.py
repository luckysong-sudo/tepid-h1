"""Tests for model layer improvements."""

import pytest
import torch

from tepid_h1.config import TepidH1Config
from tepid_h1.modeling.layers import (
    GatedDeltaMemoryEager,
    GatedDeltaMemoryReference,
    RMSNorm,
)


class TestLayerImprovements:
    """Test refactored layer code."""

    def _make_config(self) -> TepidH1Config:
        return TepidH1Config(
            hidden_size=128,
            num_layers=8,
            num_kv_heads=2,
            num_query_heads=4,
            head_dim=32,
            vocab_size=1000,
            max_position_embeddings=512,
            local_window=32,
            initializer_range=0.02,
            tie_word_embeddings=False,
            dropout=0.0,
            rotary_theta=10000.0,
            moe_num_experts=4,
            moe_top_k=2,
            moe_expert_intermediate_size=128,
            moe_shared_intermediate_size=64,
            dense_intermediate_size=128,
        )

    def test_delta_state_validation_with_valid_state(self):
        """_prepare_delta_state should accept correctly shaped state."""
        config = self._make_config()
        model = GatedDeltaMemoryReference(config)
        x = torch.randn(2, 4, config.hidden_size)
        state = torch.zeros(2, 4, 32, 32)  # matches config num_query_heads=4, head_dim=32
        validated = model._prepare_delta_state(config, x, state)
        assert validated.shape == (2, 4, 32, 32)

    def test_delta_state_validation_rejects_wrong_shape(self):
        """_prepare_delta_state should reject incorrectly shaped state."""
        config = self._make_config()
        model = GatedDeltaMemoryReference(config)
        x = torch.randn(2, 4, config.hidden_size)
        bad_state = torch.zeros(2, 2, 32, 32)  # wrong num_heads
        with pytest.raises(ValueError, match="delta state shape must be"):
            model._prepare_delta_state(config, x, bad_state)

    def test_delta_state_validation_accepts_none(self):
        """_prepare_delta_state should return initialized state when None."""
        config = self._make_config()
        model = GatedDeltaMemoryReference(config)
        x = torch.randn(2, 4, config.hidden_size)
        state = model._prepare_delta_state(config, x, None)
        assert state.shape == (2, config.num_query_heads, config.head_dim, config.head_dim)

    def test_delta_memory_forward_with_none_state(self):
        """GatedDeltaMemoryReference should initialize state when None."""
        config = self._make_config()
        model = GatedDeltaMemoryReference(config)
        x = torch.randn(2, 8, config.hidden_size)
        output, state = model(x)
        assert output.shape == (2, 8, config.hidden_size)
        assert state is not None
        assert state.shape == (2, config.num_query_heads, config.head_dim, config.head_dim)

    def test_eager_delta_forward_consistency(self):
        """GatedDeltaMemoryEager should produce same shapes as reference."""
        config = self._make_config()
        ref = GatedDeltaMemoryReference(config)
        eager = GatedDeltaMemoryEager(config)
        x = torch.randn(2, 4, config.hidden_size)

        # Both should init their own states
        out_ref, state_ref = ref(x)
        out_eager, state_eager = eager(x)

        assert out_ref.shape == out_eager.shape
        assert state_ref.shape == state_eager.shape

    def test_rmsnorm_output_shape(self):
        """RMSNorm should preserve input shape."""
        norm = RMSNorm(32, 1e-6)
        x = torch.randn(2, 8, 32)
        out = norm(x)
        assert out.shape == x.shape


class TestModelTypeAnnotations:
    """Test improved type annotations in model."""

    def _make_config(self) -> TepidH1Config:
        return TepidH1Config(
            hidden_size=128,
            num_layers=8,
            num_kv_heads=2,
            num_query_heads=4,
            head_dim=32,
            vocab_size=1000,
            max_position_embeddings=512,
            local_window=32,
            initializer_range=0.02,
            tie_word_embeddings=False,
            dropout=0.0,
            rotary_theta=10000.0,
            moe_num_experts=4,
            moe_top_k=2,
            moe_expert_intermediate_size=128,
            moe_shared_intermediate_size=64,
            dense_intermediate_size=128,
        )

    def test_causal_lm_loss_raises_on_shape_mismatch(self):
        """Loss calculation should raise on label/input shape mismatch."""
        from tepid_h1.modeling.model import _causal_lm_loss

        logits = torch.randn(2, 10, 100)
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 9))  # wrong shape
        with pytest.raises(ValueError, match="labels must have the same shape"):
            _causal_lm_loss(logits, input_ids, labels, 100)

    def test_causal_lm_loss_raises_on_short_sequence(self):
        """Loss calculation should raise on sequences < 2 tokens."""
        from tepid_h1.modeling.model import _causal_lm_loss

        logits = torch.randn(2, 1, 100)
        input_ids = torch.randint(0, 100, (2, 1))
        labels = torch.randint(0, 100, (2, 1))
        with pytest.raises(ValueError, match="requires at least two tokens"):
            _causal_lm_loss(logits, input_ids, labels, 100)

    def test_input_validation_in_forward(self):
        """TepidH1Model forward should validate input dimensions."""
        from tepid_h1.modeling.model import TepidH1Model

        config = self._make_config()
        model = TepidH1Model(config)
        # 1D input should fail
        x = torch.randn(
            10,
        )
        with pytest.raises(ValueError, match="input_ids must have shape"):
            model(x)
