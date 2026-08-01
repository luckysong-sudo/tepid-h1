"""Tests for mixed precision training utilities."""

import pytest
import torch


class TestPrecisionMode:
    """Test precision mode enumeration."""

    def test_enum_values(self):
        from tepid_h1.mixed_precision import PrecisionMode

        assert PrecisionMode.FP32.value == "fp32"
        assert PrecisionMode.BF16.value == "bfloat16"
        assert PrecisionMode.FP16.value == "float16"
        assert PrecisionMode.AUTO.value == "auto"


class TestMixedPrecisionConfig:
    """Test mixed precision configuration."""

    def test_default_config(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig

        config = MixedPrecisionConfig()
        assert config.enabled is True
        assert config.mode == "bfloat16"
        assert config.grad_scaler is True
        assert config.autocast_dtype == torch.bfloat16

    def test_fp16_config(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, PrecisionMode

        config = MixedPrecisionConfig(mode=PrecisionMode.FP16)
        assert config.autocast_dtype == torch.float16

    def test_fp32_config(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, PrecisionMode

        config = MixedPrecisionConfig(mode=PrecisionMode.FP32)
        assert config.autocast_dtype == torch.float32

    def test_disabled_config(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig

        config = MixedPrecisionConfig(enabled=False)
        assert config.enabled is False

    def test_accepts_string_mode_and_rejects_invalid_controls(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, PrecisionMode

        config = MixedPrecisionConfig(mode="float16")
        assert config.mode == PrecisionMode.FP16
        assert config.autocast_dtype == torch.float16

        with pytest.raises(ValueError, match="PrecisionMode"):
            MixedPrecisionConfig(mode="fp8")
        with pytest.raises(TypeError, match="enabled"):
            MixedPrecisionConfig(enabled=1)
        with pytest.raises(TypeError, match="grad_scaler"):
            MixedPrecisionConfig(grad_scaler=1)
        with pytest.raises(ValueError, match="autocast_dtype"):
            MixedPrecisionConfig(autocast_dtype=torch.float64)

    def test_auto_mode_uses_fp32_without_cuda(self, monkeypatch):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, PrecisionMode

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        config = MixedPrecisionConfig(mode=PrecisionMode.AUTO)

        assert config.autocast_dtype == torch.float32


class TestMixedPrecisionManager:
    """Test mixed precision manager."""

    def test_init_with_config(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)
        assert manager.config == config
        if not torch.cuda.is_available():
            assert manager.scaler is None

    def test_autocast_context_disabled(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)

        with manager.autocast_context():
            pass  # Should not raise

    def test_scale_loss(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)

        loss = torch.tensor(0.5)
        scaled = manager.scale_loss(loss)
        assert scaled == loss

    def test_step_calls_optimizer(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Should not raise
        manager.step(optimizer)

    def test_to_device_preserves_integer_token_tensors(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(mode="float16")
        manager = MixedPrecisionManager(config)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        moved = manager.to_device(input_ids, torch.device("cpu"))

        assert moved.dtype == torch.long
        assert torch.equal(moved, input_ids)

    def test_to_device_casts_floating_tensors_to_autocast_dtype(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(mode="bfloat16")
        manager = MixedPrecisionManager(config)
        activations = torch.ones(2, 3, dtype=torch.float32)

        moved = manager.to_device(activations, torch.device("cpu"))

        assert moved.dtype == torch.bfloat16

    def test_load_state_rebuilds_config_and_autocast_dtype(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        manager = MixedPrecisionManager(MixedPrecisionConfig(mode="bfloat16"))

        manager.load_state_dict(
            {
                "config": {
                    "enabled": True,
                    "mode": "float16",
                    "grad_scaler": False,
                }
            }
        )

        assert manager.config.mode == "float16"
        assert manager.config.autocast_dtype == torch.float16
        assert manager.scaler is None

    def test_load_state_rejects_invalid_config_payload(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        manager = MixedPrecisionManager(MixedPrecisionConfig())

        with pytest.raises(TypeError, match="config state"):
            manager.load_state_dict({"config": []})
