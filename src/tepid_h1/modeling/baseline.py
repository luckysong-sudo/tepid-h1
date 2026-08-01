from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from tepid_h1.config import TepidH1Config

from .layers import AttentionState, GQAAttentionNative, RMSNorm, SwiGLU
from .model import TepidH1Output, _causal_lm_loss


@dataclass(frozen=True)
class TransformerBaselineConfig:
    model: TepidH1Config
    intermediate_size: int

    @classmethod
    def active_parameter_matched(cls, model: TepidH1Config) -> TransformerBaselineConfig:
        target = hybrid_parameter_estimate(model)["active_parameters"]
        fixed = _common_parameters(model) + model.num_layers * _attention_parameters(model)
        parameters_per_intermediate = model.num_layers * 3 * model.hidden_size
        intermediate_size = max(1, round((target - fixed) / parameters_per_intermediate))
        return cls(model=model, intermediate_size=intermediate_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "full_gqa_dense_swiglu",
            "intermediate_size": self.intermediate_size,
            "model": self.model.to_dict(),
        }


def hybrid_parameter_estimate(config: TepidH1Config) -> dict[str, int]:
    counts = config.module_counts()
    delta_parameters = 7 * config.hidden_size**2
    attention_parameters = _attention_parameters(config)
    sequence = (
        counts["delta"] * delta_parameters
        + (counts["local_attention"] + counts["global_sparse_attention"]) * attention_parameters
    )
    dense = counts["dense"] * 3 * config.hidden_size * config.dense_intermediate_size
    moe_active_per_layer = (
        config.hidden_size * config.moe_num_experts
        + config.moe_top_k * 3 * config.hidden_size * config.moe_expert_intermediate_size
        + 3 * config.hidden_size * config.moe_shared_intermediate_size
    )
    moe_physical_per_layer = (
        config.hidden_size * config.moe_num_experts
        + config.moe_num_experts * 3 * config.hidden_size * config.moe_expert_intermediate_size
        + 3 * config.hidden_size * config.moe_shared_intermediate_size
    )
    common = _common_parameters(config)
    return {
        "common_parameters": common,
        "sequence_parameters": sequence,
        "channel_active_parameters": dense + counts["moe"] * moe_active_per_layer,
        "channel_physical_parameters": dense + counts["moe"] * moe_physical_per_layer,
        "active_parameters": common + sequence + dense + counts["moe"] * moe_active_per_layer,
        "physical_parameters": common + sequence + dense + counts["moe"] * moe_physical_per_layer,
    }


def baseline_parameter_estimate(config: TransformerBaselineConfig) -> dict[str, int | float]:
    model = config.model
    common = _common_parameters(model)
    sequence = model.num_layers * _attention_parameters(model)
    channel = model.num_layers * 3 * model.hidden_size * config.intermediate_size
    active = common + sequence + channel
    target = hybrid_parameter_estimate(model)["active_parameters"]
    return {
        "common_parameters": common,
        "sequence_parameters": sequence,
        "channel_parameters": channel,
        "active_parameters": active,
        "physical_parameters": active,
        "target_hybrid_active_parameters": target,
        "active_parameter_gap": active - target,
        "active_parameter_gap_percent": (active - target) / target * 100,
    }


def comparison_report(config: TepidH1Config) -> dict[str, Any]:
    baseline_config = TransformerBaselineConfig.active_parameter_matched(config)
    return {
        "schema_version": 1,
        "matching_basis": "per-token active-parameter proxy",
        "limitations": [
            "This is not a measured FLOP, latency or memory match.",
            "MoE counts router, shared expert and top-k selected expert parameters as active.",
        ],
        "hybrid": hybrid_parameter_estimate(config),
        "baseline": {
            "config": baseline_config.to_dict(),
            **baseline_parameter_estimate(baseline_config),
        },
    }


def _attention_parameters(config: TepidH1Config) -> int:
    kv_width = config.num_kv_heads * config.head_dim
    return 2 * config.hidden_size**2 + 2 * config.hidden_size * kv_width


def _common_parameters(config: TepidH1Config) -> int:
    embeddings = config.vocab_size * config.hidden_size
    output_head = 0 if config.tie_word_embeddings else embeddings
    norms_and_scales = config.num_layers * (2 * config.hidden_size + 2) + config.hidden_size
    return embeddings + output_head + norms_and_scales


class TransformerBaselineBlock(nn.Module):
    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        model = config.model
        self.sequence_norm = RMSNorm(model.hidden_size, model.rms_norm_eps)
        self.channel_norm = RMSNorm(model.hidden_size, model.rms_norm_eps)
        self.attention = GQAAttentionNative(model, local_window=None)
        self.channel = SwiGLU(model.hidden_size, config.intermediate_size)
        self.alpha = nn.Parameter(torch.full((), 0.1))
        self.beta = nn.Parameter(torch.full((), 0.1))

    def forward(
        self,
        x: Tensor,
        state: AttentionState | None = None,
    ) -> tuple[Tensor, AttentionState]:
        attended, next_state = self.attention(self.sequence_norm(x), state)
        x = x + self.alpha.tanh() * attended
        x = x + self.beta.tanh() * self.channel(self.channel_norm(x))
        return x, next_state


class TransformerBaselineModel(nn.Module):
    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        self.baseline_config = config
        self.config = config.model
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBaselineBlock(config) for _ in range(self.config.num_layers)
        )
        self.final_norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_states: tuple[AttentionState, ...] | None = None,
    ) -> TepidH1Output:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        provided_states = iter(attention_states or ())
        x = self.token_embeddings(input_ids)
        next_states: list[AttentionState] = []
        for layer in self.layers:
            x, next_state = layer(x, next(provided_states, None))
            next_states.append(next_state)
        return TepidH1Output(
            last_hidden_state=self.final_norm(x),
            delta_states=(),
            attention_states=tuple(next_states),
        )


class TransformerBaselineCausalLM(nn.Module):
    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        self.baseline_config = config
        self.config = config.model
        self.model = TransformerBaselineModel(config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.token_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_states: tuple[AttentionState, ...] | None = None,
        labels: Tensor | None = None,
    ) -> TepidH1Output:
        output = self.model(input_ids, attention_states=attention_states)
        output.logits = self.lm_head(output.last_hidden_state)
        if labels is not None:
            output.language_model_loss = _causal_lm_loss(
                output.logits,
                input_ids,
                labels,
                self.config.vocab_size,
            )
            output.loss = output.language_model_loss
        return output
