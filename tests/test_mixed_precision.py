"""Tests for mixed precision training utilities."""
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


class TestMixedPrecisionManager:
    """Test mixed precision manager."""

    def test_init_with_config(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)
        assert manager.config == config

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