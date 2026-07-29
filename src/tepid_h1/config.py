from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class SequenceMixer(str, Enum):
    DELTA = "delta"
    LOCAL_ATTENTION = "local_attention"
    GLOBAL_SPARSE_ATTENTION = "global_sparse_attention"


class ChannelMixer(str, Enum):
    DENSE = "dense"
    MOE = "moe"


@dataclass(frozen=True)
class LayerSpec:
    index: int
    macro_block: int
    sequence: SequenceMixer
    channel: ChannelMixer


MACRO_PATTERN: tuple[tuple[SequenceMixer, ChannelMixer], ...] = (
    (SequenceMixer.DELTA, ChannelMixer.DENSE),
    (SequenceMixer.DELTA, ChannelMixer.MOE),
    (SequenceMixer.LOCAL_ATTENTION, ChannelMixer.DENSE),
    (SequenceMixer.DELTA, ChannelMixer.DENSE),
    (SequenceMixer.DELTA, ChannelMixer.DENSE),
    (SequenceMixer.DELTA, ChannelMixer.MOE),
    (SequenceMixer.LOCAL_ATTENTION, ChannelMixer.DENSE),
    (SequenceMixer.GLOBAL_SPARSE_ATTENTION, ChannelMixer.DENSE),
)


@dataclass(frozen=True)
class TepidH1Config:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    local_window: int
    dense_intermediate_size: int
    moe_num_experts: int
    moe_top_k: int
    moe_expert_intermediate_size: int
    moe_shared_intermediate_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    global_reference_max_tokens: int = 2048
    rotary_theta: float = 10_000.0

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.num_layers <= 0 or self.num_layers % len(MACRO_PATTERN):
            errors.append("num_layers must be a positive multiple of 8")
        if self.hidden_size != self.num_query_heads * self.head_dim:
            errors.append("hidden_size must equal num_query_heads * head_dim")
        if self.num_query_heads % self.num_kv_heads:
            errors.append("num_query_heads must be divisible by num_kv_heads")
        if self.head_dim % 2:
            errors.append("head_dim must be even for rotary position encoding")
        if not 1 <= self.moe_top_k <= self.moe_num_experts:
            errors.append("moe_top_k must be between 1 and moe_num_experts")
        if self.local_window <= 0:
            errors.append("local_window must be positive")
        if self.max_position_embeddings < self.local_window:
            errors.append("max_position_embeddings must be >= local_window")
        if not 0.0 <= self.dropout < 1.0:
            errors.append("dropout must be in [0, 1)")
        if self.rotary_theta <= 0:
            errors.append("rotary_theta must be positive")
        if errors:
            raise ValueError("; ".join(errors))

    @property
    def layer_plan(self) -> tuple[LayerSpec, ...]:
        return tuple(
            LayerSpec(
                index=index,
                macro_block=index // len(MACRO_PATTERN),
                sequence=MACRO_PATTERN[index % len(MACRO_PATTERN)][0],
                channel=MACRO_PATTERN[index % len(MACRO_PATTERN)][1],
            )
            for index in range(self.num_layers)
        )

    def module_counts(self) -> dict[str, int]:
        counts = {member.value: 0 for member in (*SequenceMixer, *ChannelMixer)}
        for layer in self.layer_plan:
            counts[layer.sequence.value] += 1
            counts[layer.channel.value] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def prototype(cls) -> TepidH1Config:
        """Small CPU-friendly shape for API and correctness tests."""
        return cls(
            vocab_size=4096,
            hidden_size=256,
            num_layers=8,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=64,
            local_window=128,
            dense_intermediate_size=768,
            moe_num_experts=4,
            moe_top_k=2,
            moe_expert_intermediate_size=256,
            moe_shared_intermediate_size=768,
            max_position_embeddings=2048,
            global_reference_max_tokens=512,
        )

    @classmethod
    def smoke(cls) -> TepidH1Config:
        """Tiny all-module shape for training and checkpoint smoke tests."""
        return cls(
            vocab_size=128,
            hidden_size=32,
            num_layers=8,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            local_window=16,
            dense_intermediate_size=64,
            moe_num_experts=4,
            moe_top_k=2,
            moe_expert_intermediate_size=32,
            moe_shared_intermediate_size=64,
            max_position_embeddings=64,
            global_reference_max_tokens=64,
        )

    @classmethod
    def reference_28b_a7b(cls) -> TepidH1Config:
        return cls(
            vocab_size=81920,
            hidden_size=3072,
            num_layers=48,
            num_query_heads=24,
            num_kv_heads=8,
            head_dim=128,
            local_window=4096,
            dense_intermediate_size=8192,
            moe_num_experts=64,
            moe_top_k=4,
            moe_expert_intermediate_size=3072,
            moe_shared_intermediate_size=8192,
            max_position_embeddings=32768,
            global_reference_max_tokens=2048,
        )
