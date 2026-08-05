"""Tests for model quantization."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from tepid_h1.quantization import (
    QuantizationConfig,
    QuantizationMode,
    QuantizedLayer,
    apply_quantized_model,
    estimate_quantized_size,
    quantize_model,
    save_quantized_model,
)


class TestQuantizationMode:
    def test_all_modes(self) -> None:
        modes = [m.value for m in QuantizationMode]
        assert "int8" in modes
        assert "int4" in modes
        assert "nf4" in modes
        assert "fp8" in modes
        assert "fp16" in modes
        assert "bfloat16" in modes


class TestQuantizationConfig:
    def test_default_config(self) -> None:
        cfg = QuantizationConfig()
        assert cfg.mode == QuantizationMode.INT8
        assert cfg.group_size == 128
        assert cfg.symmetric is True
        assert cfg.axis == -1

    def test_custom_config(self) -> None:
        cfg = QuantizationConfig(mode=QuantizationMode.INT4, group_size=64, symmetric=False)
        assert cfg.mode == QuantizationMode.INT4
        assert cfg.group_size == 64

    def test_nf4_group_size_must_be_zero(self) -> None:
        with pytest.raises(ValueError, match="group_size"):
            QuantizationConfig(mode=QuantizationMode.NF4, group_size=128)

    def test_nf4_requires_symmetric(self) -> None:
        with pytest.raises(ValueError, match="symmetric"):
            QuantizationConfig(mode=QuantizationMode.NF4, symmetric=False)

    def test_int8_invalid_group_size(self) -> None:
        with pytest.raises(ValueError, match="group_size"):
            QuantizationConfig(mode=QuantizationMode.INT8, group_size=0)


class TestQuantizedLayer:
    def test_from_linear(self) -> None:
        linear = nn.Linear(8, 4, bias=True)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        ql = QuantizedLayer.from_linear(linear, cfg)
        assert ql.weight.shape == (4, 8)
        assert ql.bias is not None
        assert ql.original_dtype == torch.float32
        assert ql.quantization_mode == QuantizationMode.INT8

    def test_from_linear_no_bias(self) -> None:
        linear = nn.Linear(8, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        ql = QuantizedLayer.from_linear(linear, cfg)
        assert ql.bias is None

    def test_dequantize(self) -> None:
        linear = nn.Linear(8, 4, bias=True)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        ql = QuantizedLayer.from_linear(linear, cfg)
        deq = ql.dequantize()
        assert deq.shape == (4, 8)
        assert deq.dtype == torch.float32

    def test_dequantize_cached(self) -> None:
        linear = nn.Linear(8, 4, bias=True)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        ql = QuantizedLayer.from_linear(linear, cfg)
        d1 = ql.dequantize()
        d2 = ql.dequantize()
        assert d1 is d2

    def test_to_device(self) -> None:
        linear = nn.Linear(8, 4, bias=True)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        ql = QuantizedLayer.from_linear(linear, cfg)
        ql_cpu = ql.to(torch.device("cpu"))
        assert ql_cpu.device == torch.device("cpu")

    def test_nf4_round_trip_preserves_shape_and_scale(self) -> None:
        linear = nn.Linear(7, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.NF4, group_size=0)

        quantized = QuantizedLayer.from_linear(linear, cfg)
        dequantized = quantized.dequantize()

        assert quantized.weight.dtype == torch.uint8
        assert quantized.weight.shape == (4, 4)
        assert dequantized.shape == linear.weight.shape
        assert torch.allclose(dequantized.abs().amax(dim=-1), linear.weight.abs().amax(dim=-1), atol=1e-6)

    def test_int4_round_trip_preserves_original_shape(self) -> None:
        linear = nn.Linear(7, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.INT4, group_size=4)

        quantized = QuantizedLayer.from_linear(linear, cfg)
        dequantized = quantized.dequantize()

        assert quantized.weight.shape == (4, 4)
        assert dequantized.shape == linear.weight.shape


class TestQuantizeModel:
    def test_quantize_simple_model(self) -> None:
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        result = quantize_model(model, cfg)
        assert "0" in result
        assert "2" in result

    def test_quantize_with_skip(self) -> None:
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        result = quantize_model(model, cfg, skip_layers={"0"})
        assert "0" not in result
        assert "2" in result


class TestEstimateQuantizedSize:
    def test_estimate_int8(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        sizes = estimate_quantized_size(model, cfg)
        assert sizes["weight_bytes"] > 0

    def test_estimate_int4(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.INT4, symmetric=True)
        sizes = estimate_quantized_size(model, cfg)
        assert sizes["weight_bytes"] > 0

    def test_estimate_fp16(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.FP16)
        sizes = estimate_quantized_size(model, cfg)
        assert sizes["weight_bytes"] > 0


class TestApplyQuantizedModel:
    def test_apply(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        qlayers = quantize_model(model, cfg)
        apply_quantized_model(model, qlayers)
        assert model.weight.shape == (4, 8)


class TestSaveQuantizedModel:
    def test_save(self, tmp_path: Path) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        qlayers = quantize_model(model, cfg)
        path = tmp_path / "model.pt"
        artifacts = save_quantized_model(model, qlayers, path, config=cfg)
        assert path.exists()
        assert "schema_version" in artifacts
        assert "quantized_layers" in artifacts

    def test_save_nf4_model(self, tmp_path: Path) -> None:
        model = nn.Linear(8, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.NF4, group_size=0)
        qlayers = quantize_model(model, cfg)
        path = tmp_path / "nf4_model.pt"
        artifacts = save_quantized_model(model, qlayers, path, config=cfg)
        assert path.exists()
        assert artifacts["quantization_mode"] == "nf4"

    def test_estimate_nf4_size(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.NF4, group_size=0)
        sizes = estimate_quantized_size(model, cfg)
        assert sizes["weight_bytes"] > 0
        assert sizes["scale_bytes"] > 0

    def test_estimate_fp8_size(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.FP8)
        sizes = estimate_quantized_size(model, cfg)
        assert sizes["weight_bytes"] > 0

    def test_estimate_bf16_size(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.BF16)
        sizes = estimate_quantized_size(model, cfg)
        assert sizes["weight_bytes"] > 0

    def test_fp8_quantization(self) -> None:
        linear = nn.Linear(8, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.FP8, group_size=128)
        ql = QuantizedLayer.from_linear(linear, cfg)
        assert ql.weight.dtype == torch.float8_e4m3fn
        assert ql.quantization_mode == QuantizationMode.FP8

    def test_fp16_quantization(self) -> None:
        linear = nn.Linear(8, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.FP16)
        ql = QuantizedLayer.from_linear(linear, cfg)
        assert ql.weight.dtype == torch.float16
        assert ql.quantization_mode == QuantizationMode.FP16

    def test_bf16_quantization(self) -> None:
        linear = nn.Linear(8, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.BF16)
        ql = QuantizedLayer.from_linear(linear, cfg)
        assert ql.weight.dtype == torch.bfloat16
        assert ql.quantization_mode == QuantizationMode.BF16

    def test_estimate_int8_asymmetric_size(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=False)
        sizes = estimate_quantized_size(model, cfg)
        assert sizes["zero_bytes"] > 0

    def test_quantize_model_with_digit_layer_name(self) -> None:
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        result = quantize_model(model, cfg, skip_layers={"1"})
        assert "1" not in result
        assert "0" in result
        assert "2" in result

    def test_scale_and_quantize_without_zeros(self) -> None:
        from tepid_h1.quantization import _scale_and_quantize
        import torch

        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        scales = torch.tensor([0.03])
        quantized = _scale_and_quantize(weight, scales, None, -128, 127)
        assert quantized.shape == weight.shape

    def test_scale_and_quantize_with_zeros(self) -> None:
        from tepid_h1.quantization import _scale_and_quantize
        import torch

        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        scales = torch.tensor([0.03])
        zeros = torch.tensor([0.5])
        quantized = _scale_and_quantize(weight, scales, zeros, -128, 127)
        assert quantized.shape == weight.shape

    def test_per_tensor_scale_symmetric(self) -> None:
        from tepid_h1.quantization import _per_tensor_scale
        import torch

        weight = torch.randn(4, 8)
        scale = _per_tensor_scale(weight, -128, 127, symmetric=True)
        assert scale.dim() == 0
        assert scale > 0

    def test_per_tensor_scale_asymmetric(self) -> None:
        from tepid_h1.quantization import _per_tensor_scale
        import torch

        weight = torch.randn(4, 8)
        scale = _per_tensor_scale(weight, 0, 255, symmetric=False)
        assert scale.dim() == 0
        assert scale > 0

    def test_dequantize_tensor_with_zero_scales(self) -> None:
        from tepid_h1.quantization import _dequantize_tensor
        import torch

        quantized = torch.tensor([[1, 2, 3]], dtype=torch.int8)
        scales = torch.tensor([[0.01, 0.02, 0.03]])
        result = _dequantize_tensor(quantized.float(), scales, None, torch.float32, 128)
        assert result.shape == quantized.shape

    def test_estimate_quantized_size_with_bias(self) -> None:
        model = nn.Linear(8, 4, bias=True)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        sizes = estimate_quantized_size(model, cfg)
        assert "bias_bytes" in sizes
        assert sizes["bias_bytes"] == 4

    def test_estimate_quantized_size_no_bias(self) -> None:
        model = nn.Linear(8, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        sizes = estimate_quantized_size(model, cfg)
        assert "bias_bytes" not in sizes

    def test_estimate_quantized_size_int4(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.INT4, symmetric=True)
        sizes = estimate_quantized_size(model, cfg)
        # INT4 should halve the weight bytes (8*4=32 elements -> 16 bytes)
        assert sizes["weight_bytes"] == 16

    def test_estimate_quantized_size_nf4(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.NF4, group_size=0)
        sizes = estimate_quantized_size(model, cfg)
        assert sizes["weight_bytes"] > 0

    def test_apply_quantized_model_with_nested_name(self) -> None:
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        qlayers = quantize_model(model, cfg)
        apply_quantized_model(model, qlayers)
        # Should have replaced both linear layers
        assert model[0].weight.shape == (4, 8)
        assert model[2].weight.shape == (2, 4)

    def test_save_quantized_model_without_config(self, tmp_path: Path) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        qlayers = quantize_model(model, cfg)
        path = tmp_path / "model_no_config.pt"
        artifacts = save_quantized_model(model, qlayers, path, config=None)
        assert path.exists()
        assert artifacts["quantization_mode"] == "none"
        assert "config" not in artifacts

    def test_apply_quantized_model_with_nested_path(self) -> None:
        """Test apply_quantized_model with nested module paths."""
        from tepid_h1.quantization import QuantizedLayer

        inner = nn.Linear(8, 4)
        outer = nn.Sequential(inner)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        qlayers = quantize_model(outer, cfg)
        apply_quantized_model(outer, qlayers)
        assert "0" in qlayers

    def test_dequantize_tensor_with_zeros(self) -> None:
        from tepid_h1.quantization import _dequantize_tensor
        import torch

        quantized = torch.tensor([[1.0, 2.0, 3.0]])
        scales = torch.tensor([[0.01, 0.02, 0.03]])
        zeros = torch.tensor([[0.5, 0.5, 0.5]])
        result = _dequantize_tensor(quantized, scales, zeros, torch.float32, 128)
        assert result.shape == quantized.shape

    def test_dequantize_tensor_empty_last_dim(self) -> None:
        from tepid_h1.quantization import _dequantize_tensor
        import torch

        quantized = torch.empty(2, 0, dtype=torch.uint8)
        scales = torch.ones(2, 1)
        result = _dequantize_tensor(quantized, scales, None, torch.float32, 128)
        assert result.shape == (2, 0)
        assert result.dtype == torch.float32

    def test_int8_asymmetric_quantization(self) -> None:
        linear = nn.Linear(8, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=False, group_size=4)
        ql = QuantizedLayer.from_linear(linear, cfg)
        assert ql.weight.dtype == torch.int8
        assert ql.zeros is not None
        assert ql.zeros.shape == (4, 2, 1)

    def test_int4_asymmetric_quantization(self) -> None:
        linear = nn.Linear(8, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.INT4, symmetric=False, group_size=4)
        ql = QuantizedLayer.from_linear(linear, cfg)
        assert ql.weight.dtype == torch.uint8
        assert ql.zeros is not None

    def test_estimate_quantized_size_fp8(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.FP8)
        sizes = estimate_quantized_size(model, cfg)
        # FP8 should use 1 byte per weight element
        assert sizes["weight_bytes"] == 32  # 8*4 = 32

    def test_apply_quantized_model_with_bias_replacement(self) -> None:
        model = nn.Linear(8, 4, bias=True)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        qlayers = quantize_model(model, cfg)
        apply_quantized_model(model, qlayers)
        assert model.bias is not None
        assert model.bias.shape == (4,)
