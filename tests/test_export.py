"""Tests for model export utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

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
        tensors = load_file(result)
        assert "lm_head.weight" in tensors
        config_path = result.parent / "config.json"
        assert config_path.exists()
        with config_path.open() as f:
            config = json.load(f)
        assert "vocab_size" in config

    def test_export_safe_tensor_with_metadata(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "model_with_meta.safetensors"
        metadata = {"author": "test", "version": "1.0", "step": 3}
        result = exporter.export_safe_tensor(output_path, metadata=metadata)
        tensors = load_file(result)
        assert tensors
        config_path = result.parent / "config.json"
        with config_path.open() as f:
            config = json.load(f)
        assert config["author"] == "test"
        assert config["version"] == "1.0"
        assert config["step"] == 3

    def test_export_safe_tensor_rejects_metadata_config_override(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "model.safetensors"

        with pytest.raises(ValueError, match="must not override"):
            exporter.export_safe_tensor(output_path, metadata={"vocab_size": 999})

    def test_export_safe_tensor_does_not_mutate_dict_config(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(2, 2)
        config = {"vocab_size": 8, "hidden_size": 2}
        exporter = ModelExporter(model, config=config)

        exporter.export_safe_tensor(tmp_path / "model.safetensors", metadata={"author": "test"})

        assert config == {"vocab_size": 8, "hidden_size": 2}

    def test_export_safe_tensor_rejects_unsupported_config_type(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(2, 2)
        exporter = ModelExporter(model, config=object())

        with pytest.raises(ValueError, match="dataclass or dictionary"):
            exporter.export_safe_tensor(tmp_path / "model.safetensors")

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

    def test_export_for_inference_rejects_unknown_format(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="unsupported export format"):
            exporter.export_for_inference(tmp_path / "exported", formats=["safetensors", "gguf"])

    def test_export_for_inference_rejects_non_string_format(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="formats must be strings"):
            exporter.export_for_inference(
                tmp_path / "exported",
                formats=["safetensors", 1],  # type: ignore[list-item]
            )

    def test_export_for_inference_rejects_non_list_formats(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="formats must be a list"):
            exporter.export_for_inference(
                tmp_path / "exported",
                formats=("safetensors",),  # type: ignore[arg-type]
            )

    def test_export_for_inference_rejects_empty_formats(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "exported"

        with pytest.raises(ValueError, match="formats must not be empty"):
            exporter.export_for_inference(output_dir, formats=[])

        assert not output_dir.exists()

    def test_export_for_inference_deduplicates_formats(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "exported"
        exports = exporter.export_for_inference(
            output_dir,
            formats=["safetensors", "safetensors"],
        )

        assert list(exports) == ["safetensors"]
        assert exports["safetensors"].exists()

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

    @pytest.mark.parametrize(
        "input_shape, expected_error",
        [
            ((1,), "tuple of"),
            ((1, 0), "sequence_length must be positive"),
            ((True, 8), "batch_size must be an integer"),
        ],
    )
    def test_onnx_rejects_invalid_input_shape_before_export(
        self,
        exporter: ModelExporter,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        input_shape: tuple[int, ...],
        expected_error: str,
    ) -> None:
        def fail_export(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("torch.onnx.export should not run for invalid input_shape")

        monkeypatch.setattr(torch.onnx, "export", fail_export)
        output_path = tmp_path / "nested" / "model.onnx"

        with pytest.raises(ValueError, match=expected_error):
            exporter.export_onnx(output_path, input_shape=input_shape)

        assert not output_path.parent.exists()

    @pytest.mark.parametrize(
        "opset_version, expected_error",
        [
            (0, "opset_version must be positive"),
            (True, "opset_version must be an integer"),
        ],
    )
    def test_onnx_rejects_invalid_opset_before_export(
        self,
        exporter: ModelExporter,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        opset_version: int,
        expected_error: str,
    ) -> None:
        def fail_export(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("torch.onnx.export should not run for invalid opset_version")

        monkeypatch.setattr(torch.onnx, "export", fail_export)
        output_path = tmp_path / "nested" / "model.onnx"

        with pytest.raises(ValueError, match=expected_error):
            exporter.export_onnx(output_path, input_shape=(1, 8), opset_version=opset_version)

        assert not output_path.parent.exists()

    @pytest.mark.skip(reason="TorchScript not supported for keyword-only args in Python 3.14+")
    def test_export_directory_creation(self, exporter: ModelExporter, tmp_path: Path) -> None:
        output_path = tmp_path / "nested" / "dir" / "model.pt"
        result = exporter.export_torchscript(output_path)
        assert result.exists()
        assert result.parent == tmp_path / "nested" / "dir"

    def test_export_safe_tensor_creates_config(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "model.safetensors"
        result = exporter.export_safe_tensor(output_path)
        config_path = result.parent / "config.json"
        with config_path.open() as f:
            saved_config = json.load(f)
        assert saved_config["vocab_size"] == 128
        assert saved_config["hidden_size"] == 32
        assert saved_config["num_layers"] == 8

    def test_export_safe_tensor_with_nested_dir(
        self, exporter: ModelExporter, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "nested" / "dir" / "model.safetensors"
        result = exporter.export_safe_tensor(output_path)
        assert result.exists()
        assert result.parent == tmp_path / "nested" / "dir"
