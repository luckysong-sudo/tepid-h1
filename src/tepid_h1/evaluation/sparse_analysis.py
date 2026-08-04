"""Sparse attention memory estimation and block compression analysis.

This module provides analytical tools to estimate the memory savings of the
local-window plus global-anchor sparse attention pattern relative to dense
attention, and to report the block compression structure that a production
sparse kernel would need to expose.  These tools do not implement a production
sparse kernel; they produce machine-readable reports that quantify the
theoretical savings and document the block structure contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..config import TepidH1Config


@dataclass(frozen=True)
class SparseAttentionProfile:
    """Profile of a sparse attention layer's memory and block structure."""

    sequence_length: int
    local_window: int
    global_stride: int
    num_kv_heads: int
    head_dim: int
    dense_kv_elements: int
    sparse_kv_elements: int
    dense_kv_bytes: int
    sparse_kv_bytes: int
    memory_reduction_ratio: float
    local_block_count: int
    global_anchor_count: int
    total_attended_positions: int
    sparsity_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SparseAttentionReport:
    """Aggregate report across multiple sequence lengths."""

    schema_version: int
    config_summary: dict[str, Any]
    profiles: tuple[SparseAttentionProfile, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_summary": self.config_summary,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "summary": self.summary,
        }


def estimate_sparse_attention_memory(
    config: TepidH1Config,
    sequence_lengths: tuple[int, ...],
    *,
    dtype_bytes: int = 2,
) -> SparseAttentionReport:
    """Estimate KV memory savings of sparse attention vs dense attention.

    Args:
        config: Tepid-H1 model configuration.
        sequence_lengths: Sequence lengths to profile.
        dtype_bytes: Bytes per element (2 for bf16/fp16, 4 for fp32).

    Returns:
        A SparseAttentionReport with per-length profiles and aggregate summary.
    """
    if not sequence_lengths:
        raise ValueError("sequence_lengths must not be empty")
    if dtype_bytes not in {1, 2, 4, 8}:
        raise ValueError("dtype_bytes must be 1, 2, 4, or 8")

    profiles: list[SparseAttentionProfile] = []
    for seq_len in sequence_lengths:
        if seq_len <= 0:
            raise ValueError(f"sequence_length must be positive, got {seq_len}")
        profiles.append(
            _profile_single_length(
                config=config,
                sequence_length=seq_len,
                dtype_bytes=dtype_bytes,
            )
        )

    reductions = [p.memory_reduction_ratio for p in profiles]
    sparsities = [p.sparsity_ratio for p in profiles]
    summary = {
        "profile_count": len(profiles),
        "min_memory_reduction_ratio": min(reductions),
        "max_memory_reduction_ratio": max(reductions),
        "mean_memory_reduction_ratio": sum(reductions) / len(reductions),
        "min_sparsity_ratio": min(sparsities),
        "max_sparsity_ratio": max(sparsities),
        "mean_sparsity_ratio": sum(sparsities) / len(sparsities),
        "dtype_bytes": dtype_bytes,
    }

    config_summary = {
        "local_window": config.local_window,
        "global_stride": config.global_sparse_stride,
        "num_kv_heads": config.num_kv_heads,
        "head_dim": config.head_dim,
        "global_reference_max_tokens": config.global_reference_max_tokens,
    }

    return SparseAttentionReport(
        schema_version=1,
        config_summary=config_summary,
        profiles=tuple(profiles),
        summary=summary,
    )


def _profile_single_length(
    config: TepidH1Config,
    sequence_length: int,
    *,
    dtype_bytes: int,
) -> SparseAttentionProfile:
    """Profile a single sequence length."""
    local_window = config.local_window
    global_stride = config.global_sparse_stride
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim

    # Dense attention retains all KV pairs
    dense_kv_elements = sequence_length * num_kv_heads * head_dim
    dense_kv_bytes = dense_kv_elements * dtype_bytes * 2  # K and V

    # Sparse attention: for each query position, attended key positions are:
    #   - local window: up to local_window previous positions
    #   - global anchors: positions at multiples of global_stride
    # The total unique attended positions across all queries:
    local_block_count = min(local_window, sequence_length)
    global_anchor_count = _count_global_anchors(sequence_length, global_stride)

    # Total attended positions per query (union of local + global)
    # For the worst-case (last) query position:
    local_positions = min(local_window, sequence_length)
    global_positions = global_anchor_count
    # Overlap: global anchors within the local window
    overlap = _count_global_anchors(min(local_window, sequence_length), global_stride)
    total_attended = local_positions + global_positions - overlap

    # Sparse KV elements: only the attended positions need to be stored
    # In a production kernel, this would be the selected block set
    sparse_kv_elements = total_attended * num_kv_heads * head_dim
    sparse_kv_bytes = sparse_kv_elements * dtype_bytes * 2  # K and V

    memory_reduction_ratio = (
        1.0 - sparse_kv_bytes / dense_kv_bytes if dense_kv_bytes > 0 else 0.0
    )
    sparsity_ratio = (
        1.0 - total_attended / sequence_length if sequence_length > 0 else 0.0
    )

    return SparseAttentionProfile(
        sequence_length=sequence_length,
        local_window=local_window,
        global_stride=global_stride,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dense_kv_elements=dense_kv_elements,
        sparse_kv_elements=sparse_kv_elements,
        dense_kv_bytes=dense_kv_bytes,
        sparse_kv_bytes=sparse_kv_bytes,
        memory_reduction_ratio=memory_reduction_ratio,
        local_block_count=local_block_count,
        global_anchor_count=global_anchor_count,
        total_attended_positions=total_attended,
        sparsity_ratio=sparsity_ratio,
    )


def _count_global_anchors(sequence_length: int, stride: int) -> int:
    """Count how many global anchor positions exist up to sequence_length."""
    if stride <= 0:
        return 0
    return sequence_length // stride + (1 if sequence_length % stride > 0 else 0)


def describe_sparse_block_structure(
    config: TepidH1Config,
    sequence_length: int,
) -> dict[str, Any]:
    """Describe the block structure a production sparse kernel must expose.

    This function documents the compressed-block, recent-block and
    query-selected-block contract from the architecture specification,
    without implementing the kernel itself.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    local_window = config.local_window
    global_stride = config.global_sparse_stride

    # Recent (local) block: the last local_window tokens
    recent_block_start = max(0, sequence_length - local_window)
    recent_block_end = sequence_length
    recent_block_size = recent_block_end - recent_block_start

    # Compressed (global anchor) blocks: positions at multiples of stride
    anchor_positions = list(range(0, sequence_length, global_stride))

    # Query-selected blocks: for each query, the union of local + global
    # In a production kernel, this would be per-query dynamic selection
    query_selected_example = {
        "query_position": sequence_length - 1,
        "local_range": [recent_block_start, recent_block_end],
        "global_anchors": anchor_positions,
        "total_selected": recent_block_size + len(anchor_positions),
    }

    return {
        "schema_version": 1,
        "sequence_length": sequence_length,
        "block_contract": {
            "compressed_blocks": {
                "description": "Fixed-stride global anchor positions",
                "stride": global_stride,
                "positions": anchor_positions,
                "count": len(anchor_positions),
            },
            "recent_blocks": {
                "description": "Sliding window of recent tokens",
                "window_size": local_window,
                "range": [recent_block_start, recent_block_end],
                "count": recent_block_size,
            },
            "query_selected_blocks": {
                "description": "Per-query union of local and global positions",
                "example": query_selected_example,
            },
        },
        "production_kernel_status": "not_implemented",
        "reference_fallback_limit": config.global_reference_max_tokens,
    }


