from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tepid_h1.config import TepidH1Config


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
        return normalized * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


@dataclass
class MoERouterStats:
    expert_counts: Tensor
    router_probabilities: Tensor


@dataclass(frozen=True)
class AttentionState:
    """Rotated KV history and absolute position for reference streaming attention."""

    key: Tensor
    value: Tensor
    tokens_seen: int


class RoutedMoEReference(nn.Module):
    """Correctness-first Top-K MoE; replace with grouped GEMM before scale-up."""

    def __init__(self, config: TepidH1Config) -> None:
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(config.hidden_size, config.moe_expert_intermediate_size)
            for _ in range(self.num_experts)
        )
        self.shared_expert = SwiGLU(
            config.hidden_size,
            config.moe_shared_intermediate_size,
        )
        self.last_router_stats: MoERouterStats | None = None

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        probabilities = self.router(flat).softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        routed = torch.zeros_like(flat)
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        for expert_index, expert in enumerate(self.experts):
            token_indices, slots = (indices == expert_index).nonzero(as_tuple=True)
            if token_indices.numel() == 0:
                continue
            counts[expert_index] = token_indices.numel()
            expert_output = expert(flat[token_indices])
            expert_output = expert_output * weights[token_indices, slots].unsqueeze(-1)
            routed.index_add_(0, token_indices, expert_output)

        self.last_router_stats = MoERouterStats(
            expert_counts=counts.detach(),
            router_probabilities=probabilities.detach(),
        )
        return (self.shared_expert(flat) + routed).reshape(original_shape)


class GatedDeltaMemoryReference(nn.Module):
    """Sequential reference for the Gated Delta Rule-2 state orientation [K, V]."""

    def __init__(self, config: TepidH1Config) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_query_heads
        self.head_dim = config.head_dim
        projection_width = 6 * config.hidden_size
        self.in_proj = nn.Linear(config.hidden_size, projection_width, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _prepare_delta_state(config: TepidH1Config, x: Tensor, state: Tensor | None) -> Tensor:
        """Initialize or validate delta state tensor. Returns validated state."""
        batch_size = x.shape[0]
        if state is None:
            return GatedDeltaMemoryReference(config).initial_state(
                batch_size, device=x.device, dtype=x.dtype
            )
        expected = (batch_size, config.num_query_heads, config.head_dim, config.head_dim)
        if tuple(state.shape) != expected:
            raise ValueError(f"delta state shape must be {expected}, got {tuple(state.shape)}")
        return state

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch_size, sequence_length, hidden_size = x.shape
        projected = self.in_proj(x).view(
            batch_size,
            sequence_length,
            self.num_heads,
            6,
            self.head_dim,
        )
        q, k, v, raw_decay, raw_erase, raw_write = projected.unbind(dim=3)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        decay = torch.exp(-F.softplus(raw_decay))
        erase = torch.sigmoid(raw_erase)
        write = torch.sigmoid(raw_write)

        state = self._prepare_delta_state(self.config, x, state)

        outputs: list[Tensor] = []
        for step in range(sequence_length):
            key = k[:, step]
            query = q[:, step]
            value = v[:, step]

            decayed_state = decay[:, step].unsqueeze(-1) * state
            erase_key = erase[:, step] * key
            old_value = torch.einsum("bhk,bhkv->bhv", erase_key, decayed_state)
            state = decayed_state - torch.einsum("bhk,bhv->bhkv", key, old_value)
            state = state + torch.einsum(
                "bhk,bhv->bhkv",
                key,
                write[:, step] * value,
            )
            output = torch.einsum("bhk,bhkv->bhv", query, state)
            outputs.append(output)

        mixed = torch.stack(outputs, dim=1).reshape(batch_size, sequence_length, hidden_size)
        return self.out_proj(mixed), state


class GatedDeltaMemoryEager(GatedDeltaMemoryReference):
    """Algebraically fused eager Delta path with the reference state-dict layout."""

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch_size, sequence_length, hidden_size = x.shape
        projected = self.in_proj(x).view(
            batch_size,
            sequence_length,
            self.num_heads,
            6,
            self.head_dim,
        )
        q, k, v, raw_decay, raw_erase, raw_write = projected.unbind(dim=3)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        decay = torch.exp(-F.softplus(raw_decay))
        erase = torch.sigmoid(raw_erase)
        write = torch.sigmoid(raw_write)

        state = self._prepare_delta_state(self.config, x, state)

        outputs: list[Tensor] = []
        for step in range(sequence_length):
            key = k[:, step]
            decayed_state = decay[:, step].unsqueeze(-1) * state
            old_value = torch.matmul(
                (erase[:, step] * key).unsqueeze(-2),
                decayed_state,
            ).squeeze(-2)
            correction = write[:, step] * v[:, step] - old_value
            state = decayed_state + key.unsqueeze(-1) * correction.unsqueeze(-2)
            output = torch.matmul(q[:, step].unsqueeze(-2), state).squeeze(-2)
            outputs.append(output)

        mixed = torch.stack(outputs, dim=1).reshape(batch_size, sequence_length, hidden_size)
        return self.out_proj(mixed), state


class GQAAttentionReference(nn.Module):
    def __init__(self, config: TepidH1Config, *, local_window: int | None) -> None:
        super().__init__()
        self.num_query_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.local_window = local_window
        self.dropout = config.dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.rotary_theta = config.rotary_theta
        self.register_buffer(
            "rotary_inv_frequency",
            1.0
            / (
                self.rotary_theta
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            ),
            persistent=False,
        )
        q_width = self.num_query_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.out_proj = nn.Linear(q_width, config.hidden_size, bias=False)

    def _shape(self, x: Tensor, heads: int) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        return x.view(batch_size, sequence_length, heads, self.head_dim).transpose(1, 2)

    def _apply_rotary(self, x: Tensor, position_offset: int) -> Tensor:
        sequence_length = x.shape[2]
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=x.device,
            dtype=torch.float32,
        )
        inv_frequency = self.rotary_inv_frequency
        if not isinstance(inv_frequency, Tensor):
            raise TypeError("rotary_inv_frequency buffer must be a tensor")
        angles = torch.outer(positions, inv_frequency.float())
        cosine = angles.cos()[None, None].to(dtype=x.dtype)
        sine = angles.sin()[None, None].to(dtype=x.dtype)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        ).flatten(-2)

    def _attention_mask(
        self,
        query_length: int,
        key_length: int,
        past_length: int,
        device: torch.device,
    ) -> Tensor:
        query_positions = past_length + torch.arange(query_length, device=device)[:, None]
        key_positions = torch.arange(key_length, device=device)[None, :]
        mask = key_positions <= query_positions
        if self.local_window is not None:
            mask &= key_positions > query_positions - self.local_window
        return mask

    def _attend(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        """Explicit grouped-query expansion used as the correctness oracle."""
        repeat = self.num_query_heads // self.num_kv_heads
        expanded_key = key.repeat_interleave(repeat, dim=1)
        expanded_value = value.repeat_interleave(repeat, dim=1)
        return F.scaled_dot_product_attention(
            query,
            expanded_key,
            expanded_value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

    def _validate_state(self, state: AttentionState, batch_size: int) -> None:
        expected_prefix = (batch_size, self.num_kv_heads)
        if state.key.ndim != 4 or tuple(state.key.shape[:2]) != expected_prefix:
            raise ValueError(
                "attention key state must have shape "
                f"[{batch_size}, {self.num_kv_heads}, past_tokens, {self.head_dim}]"
            )
        if state.key.shape[-1] != self.head_dim:
            raise ValueError(f"attention key state head_dim must be {self.head_dim}")
        if state.value.shape != state.key.shape:
            raise ValueError("attention key and value states must have identical shapes")
        if isinstance(state.tokens_seen, bool) or not isinstance(state.tokens_seen, int):
            raise TypeError("attention state tokens_seen must be an integer")
        if state.tokens_seen < state.key.shape[2]:
            raise ValueError("attention state tokens_seen cannot be smaller than cached tokens")

    def _next_state(self, key: Tensor, value: Tensor, tokens_seen: int) -> AttentionState:
        if self.local_window is None:
            return AttentionState(key=key, value=value, tokens_seen=tokens_seen)
        retained_tokens = max(self.local_window - 1, 0)
        if retained_tokens == 0:
            return AttentionState(
                key=key[:, :, :0],
                value=value[:, :, :0],
                tokens_seen=tokens_seen,
            )
        return AttentionState(
            key=key[:, :, -retained_tokens:],
            value=value[:, :, -retained_tokens:],
            tokens_seen=tokens_seen,
        )

    def forward(
        self,
        x: Tensor,
        state: AttentionState | None = None,
    ) -> tuple[Tensor, AttentionState]:
        batch_size = x.shape[0]
        q = self._shape(self.q_proj(x), self.num_query_heads)
        current_k = self._shape(self.k_proj(x), self.num_kv_heads)
        current_v = self._shape(self.v_proj(x), self.num_kv_heads)
        past_length = 0
        position_offset = 0
        if state is not None:
            self._validate_state(state, batch_size)
            if state.key.device != x.device or state.key.dtype != x.dtype:
                raise ValueError("attention state must use the same device and dtype as the input")
            past_length = state.key.shape[2]
            position_offset = state.tokens_seen
        next_tokens_seen = position_offset + x.shape[1]
        if next_tokens_seen > self.max_position_embeddings:
            raise RuntimeError(
                "attention position exceeds max_position_embeddings: "
                f"{next_tokens_seen} > {self.max_position_embeddings}"
            )
        q = self._apply_rotary(q, position_offset)
        current_k = self._apply_rotary(current_k, position_offset)
        if state is not None:
            current_k = torch.cat((state.key, current_k), dim=2)
            current_v = torch.cat((state.value, current_v), dim=2)

        next_state = self._next_state(current_k, current_v, next_tokens_seen)
        mask = self._attention_mask(
            query_length=x.shape[1],
            key_length=current_k.shape[2],
            past_length=past_length,
            device=x.device,
        )
        attended = self._attend(q, current_k, current_v, mask)
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        return self.out_proj(attended), next_state


class GQAAttentionNative(GQAAttentionReference):
    """Native SDPA GQA path that avoids materializing repeated KV heads."""

    def _attend(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=self.num_query_heads != self.num_kv_heads,
        )


class GlobalSparseAttentionReference(GQAAttentionNative):
    """Deterministic sparse causal attention reference for global slots."""

    def __init__(self, config: TepidH1Config) -> None:
        super().__init__(config, local_window=None)
        self.max_tokens = config.global_reference_max_tokens
        self.sparse_window = config.local_window
        self.global_stride = config.global_sparse_stride

    def _attention_mask(
        self,
        query_length: int,
        key_length: int,
        past_length: int,
        device: torch.device,
    ) -> Tensor:
        query_positions = past_length + torch.arange(query_length, device=device)[:, None]
        key_positions = torch.arange(key_length, device=device)[None, :]
        causal = key_positions <= query_positions
        local = key_positions > query_positions - self.sparse_window
        global_anchor = key_positions.remainder(self.global_stride) == 0
        return causal & (local | global_anchor)

    def _validate_state(self, state: AttentionState, batch_size: int) -> None:
        super()._validate_state(state, batch_size)
        if state.tokens_seen != state.key.shape[2]:
            raise ValueError(
                "global sparse attention state must retain the complete cached history"
            )

    def forward(
        self,
        x: Tensor,
        state: AttentionState | None = None,
    ) -> tuple[Tensor, AttentionState]:
        past_tokens = 0 if state is None else state.tokens_seen
        if past_tokens + x.shape[1] > self.max_tokens:
            raise RuntimeError(
                "global sparse production kernel is not implemented; "
                f"reference fallback is limited to {self.max_tokens} total cached tokens"
            )
        return super().forward(x, state)
