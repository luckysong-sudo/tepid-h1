"""Autoregressive inference engine with KV-cache support for Tepid-H1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from .config import TepidH1Config
from .modeling import AttentionState, TepidH1CausalLM
from .modeling.cache import AttentionCache


_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


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
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
            raise TypeError("top_k must be an integer")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 1.0 <= self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in [1.0, 2.0]")
        if not isinstance(self.num_return_sequences, int) or isinstance(
            self.num_return_sequences, bool
        ):
            raise TypeError("num_return_sequences must be an integer")
        if self.num_return_sequences <= 0:
            raise ValueError("num_return_sequences must be positive")
        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be a boolean")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be float32, bfloat16 or float16")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU generation only supports float32 dtype")


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

        effective_config = _merge_generation_config(config, kwargs)

        _validate_generation_token_ids(effective_config, self._config.vocab_size)
        device, dtype = _resolve_generation_execution(effective_config)
        self.model = self.model.to(device=device, dtype=dtype)
        input_ids = _prepare_generation_input_ids(input_ids, self._config.vocab_size).to(
            device=device
        )

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

        next_logits = output.logits[:, -1, :]
        generated_ids = input_ids.clone()
        delta_states = _extract_delta_states(output)
        attention_states = _extract_attention_states(output)

        # Expand for multiple sequences
        if effective_config.num_return_sequences > 1:
            generated_ids = generated_ids.repeat_interleave(
                effective_config.num_return_sequences, dim=0
            )
            next_logits = next_logits.repeat_interleave(
                effective_config.num_return_sequences,
                dim=0,
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

        finished_sequences = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        # Autoregressive decoding
        for step in range(effective_config.max_new_tokens):
            next_token_ids = self._sample(
                next_logits,
                token_history=generated_ids,
                temperature=effective_config.temperature,
                top_k=effective_config.top_k,
                top_p=effective_config.top_p,
                repetition_penalty=effective_config.repetition_penalty,
                pad_token_id=effective_config.pad_token_id,
                eos_token_id=effective_config.eos_token_id,
                do_sample=effective_config.do_sample,
            )

            if effective_config.eos_token_id is not None:
                next_token_ids = _apply_finished_sequence_mask(
                    next_token_ids,
                    finished_sequences,
                    pad_token_id=effective_config.pad_token_id,
                    eos_token_id=effective_config.eos_token_id,
                )
                finished_sequences = finished_sequences | (
                    next_token_ids == effective_config.eos_token_id
                )

            generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)

            if finished_sequences.all():
                break
            if step == effective_config.max_new_tokens - 1:
                break

            # Advance recurrent/KV state with the token just generated.
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

        metadata = {
            "input_length": original_length,
            "output_length": generated_ids.shape[1],
            "new_tokens": generated_ids.shape[1] - original_length,
            "num_return_sequences": effective_config.num_return_sequences,
            "max_new_tokens": effective_config.max_new_tokens,
            "temperature": effective_config.temperature,
            "top_k": effective_config.top_k,
            "top_p": effective_config.top_p,
            "repetition_penalty": effective_config.repetition_penalty,
            "pad_token_id": effective_config.pad_token_id,
            "eos_token_id": effective_config.eos_token_id,
            "device": device.type,
            "dtype": str(dtype).removeprefix("torch."),
            "do_sample": effective_config.do_sample,
            "use_kv_cache": self._use_kv_cache,
        }
        return generated_ids, metadata

    def _sample(
        self,
        logits: Tensor,
        *,
        token_history: Tensor | None,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        pad_token_id: int | None,
        eos_token_id: int | None,
        do_sample: bool,
    ) -> Tensor:
        """Select the next token from logits."""
        if repetition_penalty > 1.0 and token_history is not None:
            logits = _apply_repetition_penalty(logits, token_history, repetition_penalty)

        logits = _suppress_pad_token(logits, pad_token_id, eos_token_id)

        if temperature > 0.0:
            logits = logits / temperature

        if top_k > 0:
            logits = _top_k_filter(logits, top_k)

        if 0.0 < top_p < 1.0:
            logits = _top_p_filter(logits, top_p)

        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token


def _merge_generation_config(
    config: GenerateConfig | None,
    overrides: dict[str, Any],
) -> GenerateConfig:
    base = config or GenerateConfig()
    if not overrides:
        return base

    fields = GenerateConfig.__dataclass_fields__
    unknown = sorted(set(overrides).difference(fields))
    if unknown:
        raise TypeError("unknown generation config fields: " + ", ".join(unknown))

    payload = {field: getattr(base, field) for field in fields}
    payload.update(overrides)
    return GenerateConfig(**payload)


def _prepare_generation_input_ids(input_ids: Tensor, vocab_size: int) -> Tensor:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    elif input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] <= 0:
        raise ValueError("input_ids batch size must be positive")
    if input_ids.shape[1] <= 0:
        raise ValueError("input_ids sequence length must be positive")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must use torch.long dtype")
    if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= vocab_size:
        raise ValueError(f"input_ids must contain token ids in [0, {vocab_size})")
    return input_ids


def _apply_repetition_penalty(
    logits: Tensor,
    token_history: Tensor,
    repetition_penalty: float,
) -> Tensor:
    """Apply repetition penalty to tokens already present in each sequence."""
    if logits.ndim != 2 or token_history.ndim != 2:
        return logits
    if token_history.shape[0] != logits.shape[0]:
        raise ValueError("token_history batch size must match logits")

    penalized = logits.clone()
    vocab_size = logits.shape[-1]
    for row, token_ids in enumerate(token_history):
        unique_ids = torch.unique(token_ids)
        valid_ids = unique_ids[(unique_ids >= 0) & (unique_ids < vocab_size)]
        if valid_ids.numel() == 0:
            continue
        values = penalized[row, valid_ids]
        penalized[row, valid_ids] = torch.where(
            values > 0,
            values / repetition_penalty,
            values * repetition_penalty,
        )
    return penalized


def _suppress_pad_token(
    logits: Tensor,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> Tensor:
    """Prevent padding from being generated as content."""
    if logits.ndim != 2 or pad_token_id is None or pad_token_id == eos_token_id:
        return logits
    vocab_size = logits.shape[-1]
    if not 0 <= pad_token_id < vocab_size or vocab_size <= 1:
        return logits

    filtered = logits.clone()
    filtered[:, pad_token_id] = float("-inf")
    return filtered


def _apply_finished_sequence_mask(
    next_token_ids: Tensor,
    finished_sequences: Tensor,
    *,
    pad_token_id: int | None,
    eos_token_id: int,
) -> Tensor:
    """Keep already-finished sequences from generating new content."""
    if next_token_ids.shape != finished_sequences.shape:
        raise ValueError("finished_sequences must match next_token_ids shape")
    if not bool(finished_sequences.any().item()):
        return next_token_ids

    fill_token_id = eos_token_id if pad_token_id is None else pad_token_id
    fill_tokens = torch.full_like(next_token_ids, fill_token_id)
    return torch.where(finished_sequences, fill_tokens, next_token_ids)


def _resolve_generation_execution(config: GenerateConfig) -> tuple[torch.device, torch.dtype]:
    """Resolve and validate the execution backend for generation."""
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if (
        config.device == "cuda"
        and config.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 was requested but the CUDA device does not support it")
    return torch.device(config.device), _DTYPES[config.dtype]


def _validate_generation_token_ids(config: GenerateConfig, vocab_size: int) -> None:
    """Validate configured special-token IDs against the model vocabulary."""
    for field in ("pad_token_id", "eos_token_id"):
        token_id = getattr(config, field)
        if token_id is None:
            continue
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError(f"{field} must be an integer")
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"{field} must be in [0, {vocab_size})")


def _top_k_filter(logits: Tensor, top_k: int) -> Tensor:
    """Zero out logits beyond top-k."""
    if top_k <= 0:
        return logits
    effective_top_k = min(top_k, logits.shape[-1])
    if effective_top_k == logits.shape[-1]:
        return logits
    indices_to_remove = logits < torch.topk(logits, effective_top_k)[0][..., -1, None]
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
    indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
    indices_to_remove[..., 0] = False
    sorted_indices_to_remove = indices_to_remove
    indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )
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
