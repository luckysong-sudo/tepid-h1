from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import ChannelMixer, SequenceMixer, TepidH1Config
from .layers import (
    AttentionState,
    GatedDeltaMemoryEager,
    GlobalSparseAttentionReference,
    GQAAttentionNative,
    RMSNorm,
    RoutedMoEReference,
    SwiGLU,
)


def _causal_lm_loss(logits: Tensor, input_ids: Tensor, labels: Tensor, vocab_size: int) -> Tensor:
    if labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids")
    if labels.shape[1] < 2:
        raise ValueError("causal language-model loss requires at least two tokens")
    return F.cross_entropy(
        logits[:, :-1].contiguous().float().view(-1, vocab_size),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


class TepidH1Block(nn.Module):
    def __init__(
        self, config: TepidH1Config, sequence: SequenceMixer, channel: ChannelMixer
    ) -> None:
        super().__init__()
        self.sequence_kind = sequence
        self.channel_kind = channel
        self.sequence_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.channel_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.alpha = nn.Parameter(torch.full((), 0.1))
        self.beta = nn.Parameter(torch.full((), 0.1))

        if sequence is SequenceMixer.DELTA:
            self.sequence_mixer: nn.Module = GatedDeltaMemoryEager(config)
        elif sequence is SequenceMixer.LOCAL_ATTENTION:
            self.sequence_mixer = GQAAttentionNative(config, local_window=config.local_window)
        else:
            self.sequence_mixer = GlobalSparseAttentionReference(config)

        if channel is ChannelMixer.DENSE:
            self.channel_mixer: nn.Module = SwiGLU(
                config.hidden_size,
                config.dense_intermediate_size,
            )
        else:
            self.channel_mixer = RoutedMoEReference(config)

    def forward(
        self,
        x: Tensor,
        *,
        delta_state: Tensor | None = None,
        attention_state: AttentionState | None = None,
    ) -> tuple[Tensor, Tensor | None, AttentionState | None, Tensor | None]:
        normalized = self.sequence_norm(x)
        next_delta_state: Tensor | None = None
        next_attention_state: AttentionState | None = None
        if self.sequence_kind is SequenceMixer.DELTA:
            if attention_state is not None:
                raise ValueError("Delta layers do not accept attention state")
            mixed, next_delta_state = self.sequence_mixer(normalized, delta_state)
        else:
            if delta_state is not None:
                raise ValueError("attention layers do not accept Delta state")
            mixed, next_attention_state = self.sequence_mixer(normalized, attention_state)
        x = x + self.alpha.tanh() * mixed
        channel_output = self.channel_mixer(self.channel_norm(x))
        aux_loss: Tensor | None = None
        if self.channel_kind is ChannelMixer.MOE:
            aux_loss = cast(RoutedMoEReference, self.channel_mixer).last_router_aux_loss
        x = x + self.beta.tanh() * channel_output
        return x, next_delta_state, next_attention_state, aux_loss


@dataclass
class TepidH1Output:
    last_hidden_state: Tensor
    delta_states: tuple[Tensor, ...]
    attention_states: tuple[AttentionState, ...]
    logits: Tensor | None = None
    loss: Tensor | None = None
    language_model_loss: Tensor | None = None
    aux_loss: Tensor | None = None


class TepidH1Model(nn.Module):
    def __init__(self, config: TepidH1Config) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TepidH1Block(config, layer.sequence, layer.channel) for layer in config.layer_plan
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: Tensor,
        *,
        delta_states: tuple[Tensor, ...] | None = None,
        attention_states: tuple[AttentionState, ...] | None = None,
    ) -> TepidH1Output:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        self._validate_state_counts(delta_states, attention_states)
        x = self.token_embeddings(input_ids)
        provided_delta_states = iter(delta_states or ())
        provided_attention_states = iter(attention_states or ())
        next_delta_states: list[Tensor] = []
        next_attention_states: list[AttentionState] = []
        aux_losses: list[Tensor] = []
        for layer in self.layers:
            delta_state = (
                next(provided_delta_states, None)
                if layer.sequence_kind is SequenceMixer.DELTA
                else None
            )
            attention_state = (
                next(provided_attention_states, None)
                if layer.sequence_kind is not SequenceMixer.DELTA
                else None
            )
            x, next_delta_state, next_attention_state, aux_loss = layer(
                x,
                delta_state=delta_state,
                attention_state=attention_state,
            )
            if next_delta_state is not None:
                next_delta_states.append(next_delta_state)
            if next_attention_state is not None:
                next_attention_states.append(next_attention_state)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        return TepidH1Output(
            last_hidden_state=self.final_norm(x),
            delta_states=tuple(next_delta_states),
            attention_states=tuple(next_attention_states),
            aux_loss=torch.stack(aux_losses).mean() if aux_losses else None,
        )

    def _validate_state_counts(
        self,
        delta_states: tuple[Tensor, ...] | None,
        attention_states: tuple[AttentionState, ...] | None,
    ) -> None:
        expected_delta_states = sum(
            1 for layer in self.layers if layer.sequence_kind is SequenceMixer.DELTA
        )
        expected_attention_states = len(self.layers) - expected_delta_states
        if delta_states is not None and len(delta_states) != expected_delta_states:
            raise ValueError(
                "delta_states must contain one state per Delta layer: "
                f"expected {expected_delta_states}, got {len(delta_states)}"
            )
        if attention_states is not None and len(attention_states) != expected_attention_states:
            raise ValueError(
                "attention_states must contain one state per attention layer: "
                f"expected {expected_attention_states}, got {len(attention_states)}"
            )


class TepidH1CausalLM(nn.Module):
    def __init__(self, config: TepidH1Config) -> None:
        super().__init__()
        self.config = config
        self.model = TepidH1Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.token_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        *,
        delta_states: tuple[Tensor, ...] | None = None,
        attention_states: tuple[AttentionState, ...] | None = None,
        labels: Tensor | None = None,
    ) -> TepidH1Output:
        output = self.model(
            input_ids,
            delta_states=delta_states,
            attention_states=attention_states,
        )
        output.logits = self.lm_head(output.last_hidden_state)
        if labels is not None:
            output.language_model_loss = _causal_lm_loss(
                output.logits,
                input_ids,
                labels,
                self.config.vocab_size,
            )
            output.loss = output.language_model_loss
            if output.aux_loss is not None and self.config.moe_router_aux_loss_weight:
                output.loss = (
                    output.language_model_loss
                    + self.config.moe_router_aux_loss_weight * output.aux_loss
                )
        return output
