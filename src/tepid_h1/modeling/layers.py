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

        if state is None:
            state = self.initial_state(batch_size, device=x.device, dtype=x.dtype)
        expected = (batch_size, self.num_heads, self.head_dim, self.head_dim)
        if tuple(state.shape) != expected:
            raise ValueError(f"delta state shape must be {expected}, got {tuple(state.shape)}")

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


class GQAAttentionReference(nn.Module):
    def __init__(self, config: TepidH1Config, *, local_window: int | None) -> None:
        super().__init__()
        self.num_query_heads = config.num_query_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.local_window = local_window
        self.dropout = config.dropout
        q_width = self.num_query_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.out_proj = nn.Linear(q_width, config.hidden_size, bias=False)

    def _shape(self, x: Tensor, heads: int) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        return x.view(batch_size, sequence_length, heads, self.head_dim).transpose(1, 2)

    def _attention_mask(self, sequence_length: int, device: torch.device) -> Tensor | None:
        if self.local_window is None:
            return None
        positions = torch.arange(sequence_length, device=device)
        query_positions = positions[:, None]
        key_positions = positions[None, :]
        return (key_positions <= query_positions) & (
            key_positions > query_positions - self.local_window
        )

    def forward(self, x: Tensor) -> Tensor:
        q = self._shape(self.q_proj(x), self.num_query_heads)
        k = self._shape(self.k_proj(x), self.num_kv_heads)
        v = self._shape(self.v_proj(x), self.num_kv_heads)
        repeat = self.num_query_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        mask = self._attention_mask(x.shape[1], x.device)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        attended = attended.transpose(1, 2).contiguous().view_as(x)
        return self.out_proj(attended)


class GlobalSparseAttentionReference(nn.Module):
    """Short-sequence full-attention oracle for future NSA backend comparisons."""

    def __init__(self, config: TepidH1Config) -> None:
        super().__init__()
        self.max_tokens = config.global_reference_max_tokens
        self.full_attention = GQAAttentionReference(config, local_window=None)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] > self.max_tokens:
            raise RuntimeError(
                "global sparse production kernel is not implemented; "
                f"reference fallback is limited to {self.max_tokens} tokens"
            )
        return self.full_attention(x)

