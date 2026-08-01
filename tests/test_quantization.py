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
    load_quantized_model,
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
        assert ql.weight.dtype == torch.int8
        assert ql.scales.dtype == torch.float32
        assert ql.bias is not None
        assert ql.original_dtype == torch.float32
        assert ql.quantization_mode == QuantizationMode.INT8
        assert ql.original_shape == (4, 8)

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
        assert torch.allclose(deq, linear.weight, atol=ql.scales.max().item())

    def test_quantized_weight_does_not_store_original_float_weight(self) -> None:
        linear = nn.Linear(8, 4, bias=True)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        ql = QuantizedLayer.from_linear(linear, cfg)

        assert ql.weight.dtype != linear.weight.dtype
        assert ql.weight.data_ptr() != linear.weight.data_ptr()

    def test_int8_asymmetric_uses_unsigned_storage(self) -> None:
        linear = nn.Linear(8, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=False)
        ql = QuantizedLayer.from_linear(linear, cfg)

        assert ql.weight.dtype == torch.uint8
        assert ql.zeros is not None
        assert torch.allclose(ql.dequantize(), linear.weight, atol=ql.scales.max().item())

    def test_int4_packs_last_dimension_and_dequantizes(self) -> None:
        linear = nn.Linear(7, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.INT4, group_size=4, symmetric=True)
        ql = QuantizedLayer.from_linear(linear, cfg)

        assert ql.weight.dtype == torch.uint8
        assert ql.weight.shape == (4, 4)
        assert ql.dequantize().shape == linear.weight.shape
        assert torch.allclose(ql.dequantize(), linear.weight, atol=ql.scales.max().item())

    def test_nf4_packs_weights_and_dequantizes(self) -> None:
        linear = nn.Linear(7, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.NF4, group_size=0)
        ql = QuantizedLayer.from_linear(linear, cfg)

        assert ql.weight.dtype == torch.uint8
        assert ql.weight.shape == (4, 4)
        assert ql.zeros is None
        assert ql.dequantize().shape == linear.weight.shape

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

    def test_estimate_uses_bytes_for_bias_and_totals(self) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, group_size=4)
        sizes = estimate_quantized_size(model, cfg)
        assert sizes["weight_bytes"] == 32
        assert sizes["scale_bytes"] == 32
        assert sizes["bias_bytes"] == 16
        assert sizes["total_bytes"] == 80

    def test_estimate_int4_counts_per_row_padding(self) -> None:
        model = nn.Linear(7, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.INT4, group_size=4)
        sizes = estimate_quantized_size(model, cfg)

        assert sizes["weight_bytes"] == 16


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
        layer = artifacts["quantized_layers"][""]
        assert layer["weight"].dtype == torch.int8
        assert layer["scales"].dtype == torch.float32
        assert layer["original_shape"] == (4, 8)
        assert artifacts["config"]["axis"] == -1


class TestLoadQuantizedModel:
    def test_load_saved_quantized_layers(self, tmp_path: Path) -> None:
        model = nn.Linear(7, 4, bias=False)
        cfg = QuantizationConfig(mode=QuantizationMode.INT4, group_size=4)
        qlayers = quantize_model(model, cfg)
        path = tmp_path / "model.pt"
        save_quantized_model(model, qlayers, path, config=cfg)

        loaded = load_quantized_model(path)

        assert list(loaded) == [""]
        layer = loaded[""]
        assert layer.weight.dtype == torch.uint8
        assert layer.weight.shape == (4, 4)
        assert layer.quantization_mode == QuantizationMode.INT4
        assert layer.original_shape == (4, 7)
        assert torch.allclose(layer.dequantize(), model.weight, atol=layer.scales.max().item())

    def test_loaded_layers_can_be_applied_for_inference(self, tmp_path: Path) -> None:
        torch.manual_seed(123)
        model = nn.Linear(8, 4)
        reference = nn.Linear(8, 4)
        reference.load_state_dict(model.state_dict())
        inputs = torch.randn(3, 8)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        path = tmp_path / "model.pt"
        save_quantized_model(model, quantize_model(model, cfg), path, config=cfg)

        apply_quantized_model(model, load_quantized_model(path))

        assert torch.allclose(model(inputs), reference(inputs), atol=0.02)
        assert model.weight.requires_grad is False

    def test_load_rejects_unknown_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pt"
        torch.save({"schema_version": 999, "quantized_layers": {}}, path)

        with pytest.raises(ValueError, match="schema_version"):
            load_quantized_model(path)

    def test_load_rejects_corrupt_weight_shape(self, tmp_path: Path) -> None:
        model = nn.Linear(8, 4)
        cfg = QuantizationConfig(mode=QuantizationMode.INT8, symmetric=True)
        path = tmp_path / "model.pt"
        artifact = save_quantized_model(model, quantize_model(model, cfg), path, config=cfg)
        artifact["quantized_layers"][""]["weight"] = torch.zeros(1, dtype=torch.int8)
        torch.save(artifact, path)

        with pytest.raises(ValueError, match="weight shape"):
            load_quantized_model(path)
