"""Autoregressive inference engine with KV-cache support for Tepid-H1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from .config import TepidH1Config
from .modeling import AttentionState, TepidH1CausalLM
from .modeling.cache import AttentionCache


@dataclass(frozen=True)
class GenerateConfig:
    """Configuration for autoregressive generation."""

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    num_return_sequences: int = 1
    do_sample: bool = True
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0.0 < self.temperature:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if not 0 <= self.top_k < self.max_new_tokens:
            raise ValueError("top_k must be non-negative and less than max_new_tokens")
        if not 1.0 <= self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in [1.0, 2.0]")
        if self.num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be positive")


class InferenceEngine:
    """KV-cache aware autoregressive inference engine."""

    def __init__(
        self,
        model: TepidH1CausalLM,
        config: TepidH1Config | None = None,
        *,
        use_kv_cache: bool = True,
    ) -> None:
        self.model = model.eval()
        self._use_kv_cache = use_kv_cache
        self._config = config or model.config
        self._delta_caches: list[AttentionCache | None] = []
        self._attention_caches: list[AttentionCache | None] = []
        self._setup_caches()

    def _setup_caches(self) -> None:
        model_config = self._config
        num_delta_layers = sum(
            1 for layer in model_config.layer_plan if layer.sequence.value == "delta"
        )
        num_attention_layers = model_config.num_layers - num_delta_layers
        self._delta_caches = [None] * num_delta_layers
        self._attention_caches = [None] * num_attention_layers

    def reset(self) -> None:
        """Reset all KV caches and state."""
        self._delta_caches = [None] * len(self._delta_caches)
        self._attention_caches = [None] * len(self._attention_caches)

    def generate(
        self,
        input_ids: Tensor,
        *,
        config: GenerateConfig | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs of shape [batch, seq_len].
            config: Generation configuration.

        Returns:
            Tuple of (generated_ids, metadata).
        """
        # Reset caches for each generation call
        self.reset()

        gen_config = config or GenerateConfig()
        kwargs_config = GenerateConfig(**kwargs)

        def _override(base, override, field):
            """Use override value only if it differs from the field's default."""
            override_default = GenerateConfig.__dataclass_fields__[field].default
            if override == override_default and base != override_default:
                return base
            return override

        effective_config = GenerateConfig(
            max_new_tokens=_override(
                gen_config.max_new_tokens, kwargs_config.max_new_tokens, "max_new_tokens"
            ),
            temperature=_override(gen_config.temperature, kwargs_config.temperature, "temperature"),
            top_k=_override(gen_config.top_k, kwargs_config.top_k, "top_k"),
            top_p=_override(gen_config.top_p, kwargs_config.top_p, "top_p"),
            repetition_penalty=_override(
                gen_config.repetition_penalty,
                kwargs_config.repetition_penalty,
                "repetition_penalty",
            ),
            pad_token_id=_override(
                gen_config.pad_token_id, kwargs_config.pad_token_id, "pad_token_id"
            ),
            eos_token_id=_override(
                gen_config.eos_token_id, kwargs_config.eos_token_id, "eos_token_id"
            ),
            num_return_sequences=_override(
                gen_config.num_return_sequences,
                kwargs_config.num_return_sequences,
                "num_return_sequences",
            ),
            do_sample=kwargs_config.do_sample
            if kwargs_config.do_sample is not None
            else gen_config.do_sample,
            device=_override(gen_config.device, kwargs_config.device, "device"),
            dtype=_override(gen_config.dtype, kwargs_config.dtype, "dtype"),
        )

        device = torch.device(effective_config.device)
        input_ids = input_ids.to(device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size = input_ids.shape[0]
        original_length = input_ids.shape[1]

        # Pre-fill with full sequence (tokens_seen=0 since this is a fresh prompt)
        with torch.no_grad():
            output = self.model(
                input_ids,
                delta_states=(
                    tuple(None if cache is None else cache.k_cache for cache in self._delta_caches)
                    if self._use_kv_cache
                    else None
                ),
                attention_states=(
                    tuple(
                        None
                        if cache is None
                        else AttentionState(
                            key=cast(Tensor, cache.k_cache),
                            value=cast(Tensor, cache.v_cache),
                            tokens_seen=0,
                        )
                        for cache in self._attention_caches
                    )
                    if self._use_kv_cache
                    else None
                ),
            )

        next_token_ids = input_ids[:, -1:]
        generated_ids = input_ids.clone()
        delta_states = _extract_delta_states(output)
        attention_states = _extract_attention_states(output)

        # Expand for multiple sequences
        if effective_config.num_return_sequences > 1:
            generated_ids = generated_ids.repeat_interleave(
                effective_config.num_return_sequences, dim=0
            )
            next_token_ids = next_token_ids.repeat_interleave(
                effective_config.num_return_sequences, dim=0
            )
            if delta_states is not None:
                delta_states = cast(
                    tuple[Tensor, ...],
                    _expand_states(delta_states, effective_config.num_return_sequences),
                )
            if attention_states is not None:
                attention_states = cast(
                    tuple[AttentionState, ...],
                    _expand_states(attention_states, effective_config.num_return_sequences),
                )
            batch_size *= effective_config.num_return_sequences

        # Autoregressive decoding
        for step in range(effective_config.max_new_tokens):
            # Ensure states are tuples (or None) for model forward
            ds = tuple(delta_states) if delta_states is not None else None
            ats = tuple(attention_states) if attention_states is not None else None
            with torch.no_grad():
                cached_output = self.model(
                    next_token_ids,
                    delta_states=ds,
                    attention_states=ats,
                )
            next_logits = cached_output.logits[:, -1, :]
            delta_states = _extract_delta_states(cached_output)
            attention_states = _extract_attention_states(cached_output)

            next_token_ids = self._sample(
                next_logits,
                temperature=effective_config.temperature,
                top_k=effective_config.top_k,
                top_p=effective_config.top_p,
                repetition_penalty=effective_config.repetition_penalty,
                pad_token_id=effective_config.pad_token_id,
                eos_token_id=effective_config.eos_token_id,
            )

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)

            if effective_config.eos_token_id is not None:
                eos_mask = next_token_ids == effective_config.eos_token_id
                if eos_mask.all():
                    break

        metadata = {
            "input_length": original_length,
            "output_length": generated_ids.shape[1],
            "new_tokens": generated_ids.shape[1] - original_length,
            "num_return_sequences": effective_config.num_return_sequences,
            "max_new_tokens": effective_config.max_new_tokens,
            "temperature": effective_config.temperature,
            "use_kv_cache": self._use_kv_cache,
        }
        return generated_ids, metadata

    def _sample(
        self,
        logits: Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        pad_token_id: int | None,
        eos_token_id: int | None,
    ) -> Tensor:
        """Sample next token from logits."""
        if repetition_penalty > 1.0:
            logits = _apply_repetition_penalty(logits, repetition_penalty)

        if top_k > 0:
            logits = _top_k_filter(logits, top_k)

        if 0.0 < top_p < 1.0:
            logits = _top_p_filter(logits, top_p)

        if temperature > 0.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token


def _apply_repetition_penalty(
    logits: Tensor,
    repetition_penalty: float,
) -> Tensor:
    """Apply repetition penalty to logits."""
    if logits.ndim != 2:
        return logits
    penalized = logits.clone()
    for token_ids in torch.unique(logits.argmax(dim=-1)):
        penalized[token_ids < token_ids + 1] = torch.where(
            penalized[token_ids < token_ids + 1] > 0,
            penalized[token_ids < token_ids + 1] / repetition_penalty,
            penalized[token_ids < token_ids + 1] * repetition_penalty,
        )
    return penalized


def _top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Zero out logits beyond top-k."""
    if top_k <= 0:
        return logits
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    filtered = logits.clone()
    filtered[indices_to_remove] = float("-inf")
    return filtered


def _top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    """Zero out logits with cumulative probability above top-p."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    indices_to_remove = cumulative_probs > top_p
    indices_to_remove[..., 0] = False
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float("-inf")
    return filtered_logits


def _extract_delta_states(output: Any) -> tuple[Tensor, ...] | None:
    """Extract Delta states from model output."""
    if not hasattr(output, "delta_states") or output.delta_states is None:
        return None
    return output.delta_states


def _extract_attention_states(output: Any) -> tuple[AttentionState, ...] | None:
    """Extract attention states from model output."""
    if not hasattr(output, "attention_states") or output.attention_states is None:
        return None
    return output.attention_states


def _expand_states(
    states: tuple[Tensor, ...] | tuple[AttentionState, ...] | None,
    repeats: int,
) -> tuple[Tensor, ...] | tuple[AttentionState, ...] | None:
    """Expand states for multiple sequences."""
    if states is None:
        return None
    expanded: list[Tensor | AttentionState] = []
    for state in states:
        if isinstance(state, AttentionState):
            expanded.append(
                AttentionState(
                    key=state.key.repeat_interleave(repeats, dim=0),
                    value=state.value.repeat_interleave(repeats, dim=0),
                    tokens_seen=state.tokens_seen,
                )
            )
        else:
            expanded.append(state.repeat_interleave(repeats, dim=0))
    if all(isinstance(item, AttentionState) for item in expanded):
        return tuple(cast(list[AttentionState], expanded))
    return tuple(cast(list[Tensor], expanded))


def decode_text(
    generated_ids: Tensor,
    input_ids: Tensor,
    tokenizer: Any,
    *,
    skip_special_tokens: bool = True,
) -> str:
    """Decode generated token IDs to text using a tokenizer.

    Args:
        generated_ids: Token IDs from generate().
        input_ids: Original input token IDs.
        tokenizer: A tokenizer with a decode() method.
        skip_special_tokens: Whether to skip special tokens.

    Returns:
        Decoded text string.
    """
    if not hasattr(tokenizer, "decode"):
        raise AttributeError(
            f"tokenizer must have a 'decode' method, got {type(tokenizer).__name__}"
        )
    generated_tokens = generated_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(generated_tokens.tolist(), skip_special_tokens=skip_special_tokens)
