"""Model export utilities for production deployment."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_model


_SUPPORTED_EXPORT_FORMATS = frozenset({"torchscript", "onnx", "safetensors"})


class ModelExporter:
    """Exports models to various formats for deployment."""

    def __init__(self, model: nn.Module, config: Any) -> None:
        self.model = model
        self.config = config

    def export_torchscript(
        self,
        output_path: Path,
        example_input: torch.Tensor | None = None,
    ) -> Path:
        """Export model to TorchScript format.

        Args:
            output_path: Path to save the TorchScript model.
            example_input: Example input for tracing.

        Returns:
            Path to the exported model.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            if example_input is not None:
                traced = torch.jit.trace(self.model, example_input)
                traced.save(str(output_path))
            else:
                scripted = torch.jit.script(self.model)
                scripted.save(str(output_path))

        return output_path

    def export_onnx(
        self,
        output_path: Path,
        input_shape: tuple[int, ...] = (1, 512),
        opset_version: int = 17,
    ) -> Path:
        """Export model to ONNX format.

        Args:
            output_path: Path to save the ONNX model.
            input_shape: Input tensor shape (batch_size, sequence_length).
            opset_version: ONNX opset version.

        Returns:
            Path to the exported model.
        """
        validated_input_shape = _validate_onnx_input_shape(input_shape)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        dummy_input = torch.randint(0, self.config.vocab_size, validated_input_shape)

        torch.onnx.export(
            self.model,
            (dummy_input,),
            str(output_path),
            opset_version=opset_version,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )

        return output_path

    def export_safe_tensor(
        self,
        output_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Export model weights to SafeTensor format.

        Args:
            output_path: Path to save the SafeTensor file.
            metadata: Additional metadata to include.

        Returns:
            Path to the exported weights.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_model(self.model, str(output_path), metadata=_safetensors_metadata(metadata))

        # Save config separately
        config_path = output_path.parent / "config.json"
        config_dict = _export_config_payload(self.config)
        if metadata:
            overlapping = sorted(set(config_dict).intersection(metadata))
            if overlapping:
                raise ValueError(
                    "export metadata must not override model config keys: " + ", ".join(overlapping)
                )
            config_dict.update(metadata)
        config_path.write_text(json.dumps(config_dict, indent=2, ensure_ascii=False))

        return output_path

    def export_for_inference(
        self,
        output_dir: Path,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Export model in multiple formats.

        Args:
            output_dir: Directory to save exported models.
            formats: List of formats to export ("torchscript", "onnx", "safetensors").

        Returns:
            Dictionary mapping format names to output paths.
        """
        export_formats = _normalize_export_formats(formats)

        output_dir.mkdir(parents=True, exist_ok=True)
        exports: dict[str, Path] = {}

        for fmt in export_formats:
            if fmt == "torchscript":
                exports["torchscript"] = self.export_torchscript(output_dir / "model.pt")
            elif fmt == "onnx":
                exports["onnx"] = self.export_onnx(output_dir / "model.onnx")
            elif fmt == "safetensors":
                exports["safetensors"] = self.export_safe_tensor(output_dir / "model.safetensors")

        return exports

    def get_export_config(self) -> dict[str, Any]:
        """Get configuration for model export.

        Returns:
            Dictionary with export configuration.
        """
        return {
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "num_query_heads": self.config.num_query_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "head_dim": self.config.head_dim,
        }


def _normalize_export_formats(formats: list[str] | None) -> list[str]:
    if formats is None:
        requested = ["torchscript", "onnx", "safetensors"]
    else:
        if not isinstance(formats, list):
            raise ValueError("export formats must be a list")
        if not formats:
            raise ValueError("export formats must not be empty")
        requested = formats

    normalized: list[str] = []
    for fmt in requested:
        if not isinstance(fmt, str):
            raise ValueError("export formats must be strings")
        if fmt not in _SUPPORTED_EXPORT_FORMATS:
            supported = ", ".join(sorted(_SUPPORTED_EXPORT_FORMATS))
            raise ValueError(f"unsupported export format {fmt!r}; supported formats: {supported}")
        if fmt not in normalized:
            normalized.append(fmt)
    return normalized


def _validate_onnx_input_shape(input_shape: tuple[int, ...]) -> tuple[int, int]:
    if not isinstance(input_shape, tuple) or len(input_shape) != 2:
        raise ValueError("input_shape must be a tuple of (batch_size, sequence_length)")

    batch_size, sequence_length = input_shape
    for name, value in (("batch_size", batch_size), ("sequence_length", sequence_length)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"input_shape {name} must be an integer")
        if value <= 0:
            raise ValueError(f"input_shape {name} must be positive")

    return batch_size, sequence_length


def _export_config_payload(config: Any) -> dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        return dict(asdict(config))
    if isinstance(config, dict):
        return dict(config)
    raise ValueError("export config must be a dataclass or dictionary")


def _safetensors_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    payload = {"format": "safetensors"}
    if metadata:
        payload.update({str(key): str(value) for key, value in metadata.items()})
    return payload
