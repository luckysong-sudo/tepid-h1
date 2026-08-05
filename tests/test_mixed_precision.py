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

    def test_to_device_cpu(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)

        tensor = torch.randn(3, 4)
        result = manager.to_device(tensor, torch.device("cpu"))
        assert result.shape == (3, 4)

    def test_state_dict_round_trip(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)
        state = manager.state_dict()
        assert state["config"]["enabled"] is True
        assert state["config"]["mode"] == "bfloat16"

    def test_load_state_dict_restores_config(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        manager = MixedPrecisionManager(MixedPrecisionConfig())
        manager.load_state_dict({"config": {"enabled": False, "mode": "fp32", "grad_scaler": True}})
        assert manager.config.enabled is False
        assert manager.config.mode.value == "fp32"

    def test_autocast_context_enabled_cpu(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=True, mode="fp16")
        manager = MixedPrecisionManager(config)

        with manager.autocast_context():
            x = torch.randn(2, 3)
            # Should run without error on CPU

    def test_autocast_context_enabled_fp32(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=True, mode="fp32")
        manager = MixedPrecisionManager(config)

        with manager.autocast_context():
            x = torch.randn(2, 3)
            assert x.dtype == torch.float32

    def test_scale_loss_returns_unscaled_when_no_scaler(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)

        loss = torch.tensor(2.0)
        result = manager.scale_loss(loss)
        assert result.item() == 2.0

    def test_unscale_grads_noop_when_no_scaler(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Should not raise
        manager.unscale_grads(optimizer)

    def test_step_calls_optimizer_step_when_no_scaler(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        initial_weight = model.weight.data.clone()

        manager.step(optimizer)

        # Weights should remain unchanged since no forward/backward was done
        assert torch.equal(model.weight.data, initial_weight)

    def test_to_device_fp16_conversion(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager, PrecisionMode

        config = MixedPrecisionConfig(enabled=True, mode=PrecisionMode.FP16)
        manager = MixedPrecisionManager(config)

        tensor = torch.randn(3, 4, dtype=torch.float32)
        result = manager.to_device(tensor, torch.device("cpu"))
        assert result.dtype == torch.float16

    def test_to_device_no_conversion_for_bf16(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager, PrecisionMode

        config = MixedPrecisionConfig(enabled=True, mode=PrecisionMode.BF16)
        manager = MixedPrecisionManager(config)

        tensor = torch.randn(3, 4, dtype=torch.float32)
        result = manager.to_device(tensor, torch.device("cpu"))
        # BF16 mode should not convert to half
        assert result.dtype == torch.float32

    def test_to_device_no_conversion_when_disabled(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)

        tensor = torch.randn(3, 4, dtype=torch.float16)
        result = manager.to_device(tensor, torch.device("cpu"))
        assert result.dtype == torch.float16

    def test_state_dict_without_scaler(self):
        from tepid_h1.mixed_precision import MixedPrecisionConfig, MixedPrecisionManager

        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)
        state = manager.state_dict()

        assert "scaler_state" not in state
        assert state["config"]["enabled"] is False