"""Tests for model export utilities."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tepid_h1.config import TepidH1Config
from tepid_h1.export import ModelExporter
from tepid_h1.modeling import TepidH1CausalLM


class TestModelExporter:
    """Tests for ModelExporter class."""

    @pytest.fixture
    def model(self) -> TepidH1CausalLM:
        config = TepidH1Config.smoke()
        return TepidH1CausalLM(config)

    @pytest.fixture
    def exporter(self, model: TepidH1CausalLM) -> ModelExporter:
        return ModelExporter(model, config=model.config)

    @pytest.mark.skip(reason="TorchScript not supported for keyword-only args in Python 3.14+")
    def test_export_torchscript(self, exporter: ModelExporter, tmp_path: Path) -> None:
        output_path = tmp_path / "model.pt"
        result = exporter.export_torchscript(output_path)
        assert result.exists()
        loaded = torch.jit.load(result)
        assert loaded is not None

    @pytest.mark.skip(reason="TorchScript tracing not supported for complex models in Python 3.14+")
    def test_export_torchscript_with_example_input(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "model_traced.pt"
        example_input = torch.randint(0, 128, (1, 8))
        result = exporter.export_torchscript(output_path, example_input=example_input)
        assert result.exists()

    @pytest.mark.skip(reason="onnxscript module not installed")
    def test_export_onnx(self, exporter: ModelExporter, tmp_path: Path) -> None:
        output_path = tmp_path / "model.onnx"
        result = exporter.export_onnx(output_path, input_shape=(1, 8))
        assert result.exists()

    def test_export_safe_tensor(self, exporter: ModelExporter, tmp_path: Path) -> None:
        output_path = tmp_path / "model.safetensors"
        result = exporter.export_safe_tensor(output_path)
        assert result.exists()
        config_path = result.parent / "config.json"
        assert config_path.exists()
        with config_path.open() as f:
            config = json.load(f)
        assert "vocab_size" in config

    def test_export_safe_tensor_with_metadata(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "model_with_meta.safetensors"
        metadata = {"author": "test", "version": "1.0"}
        result = exporter.export_safe_tensor(output_path, metadata=metadata)
        config_path = result.parent / "config.json"
        with config_path.open() as f:
            config = json.load(f)
        assert config["author"] == "test"
        assert config["version"] == "1.0"

    @pytest.mark.skip(reason="TorchScript not supported for keyword-only args in Python 3.14+")
    def test_export_for_inference(self, exporter: ModelExporter, tmp_path: Path) -> None:
        output_dir = tmp_path / "exported"
        exports = exporter.export_for_inference(output_dir, formats=["safetensors"])
        assert "safetensors" in exports
        assert exports["safetensors"].exists()

    @pytest.mark.skip(reason="TorchScript not supported for keyword-only args in Python 3.14+")
    def test_export_for_inference_default_formats(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "exported"
        exports = exporter.export_for_inference(output_dir)
        assert "torchscript" in exports
        assert "onnx" in exports
        assert "safetensors" in exports

    def test_get_export_config(self, exporter: ModelExporter) -> None:
        config = exporter.get_export_config()
        assert "vocab_size" in config
        assert "hidden_size" in config
        assert "num_layers" in config
        assert config["vocab_size"] == 128
        assert config["hidden_size"] == 32
        assert config["num_layers"] == 8

    @pytest.mark.skip(reason="TorchScript not supported for keyword-only args in Python 3.14+")
    def test_torchscript_requires_eval(self, exporter: ModelExporter, tmp_path: Path) -> None:
        exporter.model.train()
        output_path = tmp_path / "model.pt"
        exporter.export_torchscript(output_path)
        assert not exporter.model.training

    @pytest.mark.skip(reason="onnxscript module not installed")
    def test_onnx_requires_eval(self, exporter: ModelExporter, tmp_path: Path) -> None:
        exporter.model.train()
        output_path = tmp_path / "model.onnx"
        exporter.export_onnx(output_path)
        assert not exporter.model.training

    @pytest.mark.skip(reason="TorchScript not supported for keyword-only args in Python 3.14+")
    def test_export_directory_creation(self, exporter: ModelExporter, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "dir" / "model.pt"
        result = exporter.export_torchscript(output_path)
        assert result.exists()
        assert result.parent == tmp_path / "nested" / "dir"

    def test_export_safe_tensor_creates_config(self, exporter: ModelExporter, tmp_path: Path) -> None:
        output_path = tmp_path / "model.safetensors"
        result = exporter.export_safe_tensor(output_path)
        config_path = result.parent / "config.json"
        with config_path.open() as f:
            saved_config = json.load(f)
        assert saved_config["vocab_size"] == 128
        assert saved_config["hidden_size"] == 32
        assert saved_config["num_layers"] == 8

    def test_export_safe_tensor_with_nested_dir(self, exporter: ModelExporter, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "dir" / "model.safetensors"
        result = exporter.export_safe_tensor(output_path)
        assert result.exists()
        assert result.parent == tmp_path / "nested" / "dir"

    def test_export_for_inference_safetensors_only(self, exporter: ModelExporter, tmp_path: Path) -> None:
        output_dir = tmp_path / "exported"
        exports = exporter.export_for_inference(output_dir, formats=["safetensors"])
        assert "safetensors" in exports
        assert exports["safetensors"].exists()
        config_path = exports["safetensors"].parent / "config.json"
        assert config_path.exists()