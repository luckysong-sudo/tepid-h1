"""KV-cache inference optimization for Tepid-H1 model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor


@dataclass
class AttentionCache:
    """KV cache for efficient incremental attention decoding."""

    k_cache: Tensor | None = None
    v_cache: Tensor | None = None
    seq_len: int = 0
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.k_cache is not None and self.v_cache is None:
            raise ValueError("k_cache requires v_cache to be provided")
        if self.v_cache is not None and self.k_cache is None:
            raise ValueError("v_cache requires k_cache to be provided")
        if self.k_cache is not None:
            self.device = self.k_cache.device
            self.dtype = self.k_cache.dtype
            self.seq_len = self.k_cache.shape[-2]

    def to(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> "AttentionCache":
        if self.k_cache is None:
            return self
        new_device = device or self.device
        new_dtype = dtype or self.dtype
        if new_device == self.device and new_dtype == self.dtype:
            return self
        self.k_cache = self.k_cache.to(device=new_device, dtype=new_dtype)
        if self.v_cache is None:
            raise ValueError("v_cache requires k_cache to be provided")
        self.v_cache = self.v_cache.to(device=new_device, dtype=new_dtype)
        self.device = new_device
        self.dtype = new_dtype
        return self

    def update(
        self,
        k_new: Tensor,
        v_new: Tensor,
        *,
        cache_length: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if k_new.shape[-2] != v_new.shape[-2]:
            raise ValueError("key and value sequences must have equal length")
        if k_new.ndim != 3:
            raise ValueError(f"expected 3D tensors, got {k_new.ndim}D")
        if k_new.shape[0] != v_new.shape[0]:
            raise ValueError("batch dimensions must match")
        if k_new.dtype != v_new.dtype:
            raise ValueError("key and value dtypes must match")
        if cache_length is None:
            cache_length = k_new.shape[-2]
        if cache_length <= 0:
            raise ValueError("cache_length must be positive")

        if self.k_cache is None:
            self.k_cache = k_new
            self.v_cache = v_new
            self.seq_len = k_new.shape[-2]
            return k_new, v_new

        max_seq_len = self.k_cache.shape[-2] + k_new.shape[-2]
        if max_seq_len > cache_length:
            shift = max_seq_len - cache_length
            if shift >= max_seq_len:
                raise ValueError("cache_length must accommodate the full sequence")
            k_new = k_new[:, :, shift:]
            v_new = v_new[:, :, shift:]
            if k_new.shape[-2] == 0 or v_new.shape[-2] == 0:
                if self.v_cache is None:
                    raise ValueError("v_cache requires k_cache to be provided")
                return self.k_cache, self.v_cache

        prev_k = self.k_cache
        prev_v = self.v_cache
        if prev_v is None:
            raise ValueError("v_cache requires k_cache to be provided")
        new_k = torch.cat([prev_k, k_new], dim=-2)
        new_v = torch.cat([prev_v, v_new], dim=-2)
        self.k_cache = new_k
        self.v_cache = new_v
        self.seq_len = new_k.shape[-2]
        return new_k, new_v

    def reset(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0

    @property
    def is_empty(self) -> bool:
        return self.k_cache is None

    @property
    def state_dict(self) -> dict[str, Any]:
        return {
            "seq_len": self.seq_len,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "k_cache": self.k_cache if self.k_cache is None else self.k_cache.detach(),
            "v_cache": self.v_cache if self.v_cache is None else self.v_cache.detach(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("attention cache state must be a mapping")
        seq_len = state.get("seq_len", 0)
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 0:
            raise ValueError("seq_len must be a non-negative integer")
        device_str = state.get("device", "cpu")
        dtype_str = state.get("dtype", "float32")
        k_cache = state.get("k_cache")
        v_cache = state.get("v_cache")
        if k_cache is not None:
            k_cache = torch.as_tensor(k_cache)
        if v_cache is not None:
            v_cache = torch.as_tensor(v_cache)
        device = _resolve_cache_device(device_str)
        dtype = _resolve_cache_dtype(dtype_str)
        if (k_cache is None) != (v_cache is None):
            raise ValueError("cached key and value states must be provided together")
        if k_cache is None and seq_len != 0:
            raise ValueError("empty attention cache state must have seq_len 0")
        if k_cache is not None:
            _validate_cache_tensors(k_cache, v_cache, seq_len, expected_dtype=dtype)
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype
        self.k_cache = cast(Tensor | None, k_cache)
        self.v_cache = cast(Tensor | None, v_cache)


def _validate_cache_tensors(
    k_cache: Tensor,
    v_cache: Any,
    seq_len: int,
    *,
    expected_dtype: torch.dtype,
) -> None:
    if not isinstance(v_cache, Tensor):
        raise TypeError("cached value state must be a tensor")
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("cached key and value states must be 3D tensors")
    if k_cache.shape != v_cache.shape:
        raise ValueError("cached key and value states must have identical shapes")
    if k_cache.shape[-2] != seq_len:
        raise ValueError("seq_len must match cached key and value sequence length")
    if k_cache.dtype != v_cache.dtype:
        raise ValueError("cached key and value dtypes must match")
    if k_cache.dtype != expected_dtype:
        raise ValueError("attention cache dtype must match cached tensors")


def _resolve_cache_device(value: Any) -> torch.device:
    try:
        return torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("attention cache device must be a valid torch device") from error


def _resolve_cache_dtype(value: Any) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "torch.float32": torch.float32,
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
    }
    if value not in aliases:
        raise ValueError("attention cache dtype must be float32, float16 or bfloat16")
    return aliases[value]
