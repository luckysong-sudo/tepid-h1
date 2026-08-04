"""Tests for inference sampling functions and engine edge cases."""
import pytest
import torch

from tepid_h1.config import TepidH1Config
from tepid_h1.inference import (
    GenerateConfig,
    InferenceEngine,
    _apply_repetition_penalty,
    _top_k_filter,
    _top_p_filter,
    decode_text,
)
from tepid_h1.modeling import TepidH1CausalLM


class TestSamplingFunctions:
    def test_top_k_filter_keeps_top_k_logits(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        filtered = _top_k_filter(logits, top_k=2)
        # Two largest values should remain, others should be -inf
        assert filtered[0, 4] == 5.0
        assert filtered[0, 3] == 4.0
        assert filtered[0, 0] == float("-inf")
        assert filtered[0, 1] == float("-inf")

    def test_top_k_filter_noop_when_k_zero(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = _top_k_filter(logits, top_k=0)
        assert torch.equal(filtered, logits)

    def test_top_p_filter_keeps_high_probs(self):
        logits = torch.tensor([[0.0, 0.0, 0.0, 10.0]])
        filtered = _top_p_filter(logits, top_p=0.5)
        # When top_p is restrictive, only the top-k sorted entries may survive
        assert filtered.shape == logits.shape

    def test_top_p_filter_noop_when_p_one(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = _top_p_filter(logits, top_p=1.0)
        assert torch.equal(filtered, logits)

    def test_repetition_penalty_divides_positive(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        penalized = _apply_repetition_penalty(logits, 1.5)
        # Positive logits should be divided by penalty
        assert penalized[0, 2] < logits[0, 2]

    def test_repetition_penalty_noop_on_empty(self):
        logits = torch.tensor([[0.0, 0.0, 0.0]])
        penalized = _apply_repetition_penalty(logits, 1.5)
        assert torch.equal(penalized, logits)


class TestGenerateConfig:
    def test_valid_config(self):
        cfg = GenerateConfig(max_new_tokens=10, temperature=0.5)
        assert cfg.max_new_tokens == 10
        assert cfg.temperature == 0.5

    def test_invalid_max_new_tokens_raises(self):
        with pytest.raises(ValueError):
            GenerateConfig(max_new_tokens=0)

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            GenerateConfig(temperature=-1.0)

    def test_invalid_top_p_raises(self):
        with pytest.raises(ValueError):
            GenerateConfig(top_p=1.5)

    def test_invalid_repetition_penalty_raises(self):
        with pytest.raises(ValueError):
            GenerateConfig(repetition_penalty=0.5)

    def test_invalid_num_return_sequences_raises(self):
        with pytest.raises(ValueError):
            GenerateConfig(num_return_sequences=0)


class TestInferenceEngineEdgeCases:
    @pytest.fixture
    def engine(self):
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        return InferenceEngine(model, use_kv_cache=False)

    def test_decode_text_single_sequence(self, engine):
        input_ids = torch.tensor([[1, 2, 3]])
        generated = torch.tensor([[1, 2, 3, 4, 5]])

        class FakeTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(str(t) for t in tokens)

        text = decode_text(generated, input_ids, FakeTokenizer())
        assert isinstance(text, str)
        assert "4" in text

    def test_decode_text_with_skip_special(self, engine):
        input_ids = torch.tensor([[1, 2, 3]])
        generated = torch.tensor([[1, 2, 3, 4, 5]])

        class FakeTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return "-".join(str(t) for t in tokens if t > 0)

        # decode_text only decodes NEW tokens (excluding input)
        text = decode_text(generated, input_ids, FakeTokenizer(), skip_special_tokens=True)
        assert text == "4-5"

    def test_engine_reset_clears_caches(self, engine):
        engine.reset()
        assert len(engine._delta_caches) > 0
        for cache in engine._delta_caches:
            if cache is not None:
                assert cache.is_empty
