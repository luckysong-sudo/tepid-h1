from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tepid_h1.config import ChannelMixer, SequenceMixer, TepidH1Config

from .layers import (
    GQAAttentionReference,
    GatedDeltaMemoryReference,
    GlobalSparseAttentionReference,
    RMSNorm,
    RoutedMoEReference,
    SwiGLU,
)


class TepidH1Block(nn.Module):
    def __init__(self, config: TepidH1Config, sequence: SequenceMixer, channel: ChannelMixer) -> None:
        super().__init__()
        self.sequence_kind = sequence
        self.channel_kind = channel
        self.sequence_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.channel_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.alpha = nn.Parameter(torch.full((), 0.1))
        self.beta = nn.Parameter(torch.full((), 0.1))

        if sequence is SequenceMixer.DELTA:
            self.sequence_mixer: nn.Module = GatedDeltaMemoryReference(config)
        elif sequence is SequenceMixer.LOCAL_ATTENTION:
            self.sequence_mixer = GQAAttentionReference(config, local_window=config.local_window)
        else:
            self.sequence_mixer = GlobalSparseAttentionReference(config)

        if channel is ChannelMixer.DENSE:
            self.channel_mixer: nn.Module = SwiGLU(
                config.hidden_size,
                config.dense_intermediate_size,
            )
        else:
            self.channel_mixer = RoutedMoEReference(config)

    def forward(self, x: Tensor, delta_state: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        normalized = self.sequence_norm(x)
        next_state: Tensor | None = None
        if self.sequence_kind is SequenceMixer.DELTA:
            mixed, next_state = self.sequence_mixer(normalized, delta_state)
        else:
            mixed = self.sequence_mixer(normalized)
        x = x + self.alpha.tanh() * mixed
        x = x + self.beta.tanh() * self.channel_mixer(self.channel_norm(x))
        return x, next_state


@dataclass
class TepidH1Output:
    last_hidden_state: Tensor
    delta_states: tuple[Tensor, ...]
    logits: Tensor | None = None


class TepidH1Model(nn.Module):
    def __init__(self, config: TepidH1Config) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TepidH1Block(config, layer.sequence, layer.channel)
            for layer in config.layer_plan
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
    ) -> TepidH1Output:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        x = self.token_embeddings(input_ids)
        provided_states = iter(delta_states or ())
        next_states: list[Tensor] = []
        for layer in self.layers:
            state = next(provided_states, None) if layer.sequence_kind is SequenceMixer.DELTA else None
            x, next_state = layer(x, state)
            if next_state is not None:
                next_states.append(next_state)
        return TepidH1Output(
            last_hidden_state=self.final_norm(x),
            delta_states=tuple(next_states),
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
    ) -> TepidH1Output:
        output = self.model(input_ids, delta_states=delta_states)
        output.logits = self.lm_head(output.last_hidden_state)
        return output

