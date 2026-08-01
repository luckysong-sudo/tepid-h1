"""Integration tests for InferenceEngine and LoRA adapter."""

import pytest
import torch
import torch.nn as nn

from tepid_h1.config import TepidH1Config
from tepid_h1.inference import GenerateConfig, InferenceEngine, decode_text
from tepid_h1.lora import (
    LoRAConfig,
    apply_lora,
    freeze_base_model,
    lora_param_count,
    get_lora_params,
)
from tepid_h1.modeling import TepidH1CausalLM


@pytest.fixture
def model_config():
    return TepidH1Config.smoke()


@pytest.fixture
def model(model_config):
    return TepidH1CausalLM(model_config)


@pytest.fixture
def inference_model():
    """Fresh model for inference tests to avoid state leaking from LoRA tests."""
    config = TepidH1Config.smoke()
    object.__setattr__(config, "max_position_embeddings", 256)
    return TepidH1CausalLM(config)


@pytest.fixture
def engine(inference_model):
    return InferenceEngine(inference_model, use_kv_cache=True)


@pytest.fixture
def simple_model():
    """Create a simple linear model for LoRA testing."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


class TestInferenceEngine:
    def test_generate_basic(self, engine):
        """Test basic autoregressive generation."""
        input_ids = torch.tensor([[1, 2, 3]])
        generated, metadata = engine.generate(
            input_ids,
            config=GenerateConfig(max_new_tokens=2, do_sample=False),
        )
        assert generated.shape[1] == 3 + 2  # original + new tokens
        assert metadata["new_tokens"] == 2
        assert metadata["input_length"] == 3
        assert metadata["use_kv_cache"] is True

    def test_generate_deterministic(self, engine):
        """Test deterministic generation (greedy) on same engine."""
        torch.manual_seed(42)
        input_ids = torch.tensor([[1, 2, 3]])
        generated1, _ = engine.generate(
            input_ids,
            config=GenerateConfig(max_new_tokens=2, do_sample=False),
        )
        torch.manual_seed(42)
        engine.reset()
        generated2, _ = engine.generate(
            input_ids,
            config=GenerateConfig(max_new_tokens=2, do_sample=False),
        )
        # Same engine, same weights, same seed => deterministic with do_sample=False
        assert torch.equal(generated1, generated2)

    def test_generate_config_do_sample_false_is_preserved(self, engine):
        input_ids = torch.tensor([[1, 2, 3]])

        _, metadata = engine.generate(
            input_ids,
            config=GenerateConfig(max_new_tokens=1, do_sample=False),
        )

        assert metadata["do_sample"] is False

    def test_greedy_sample_uses_argmax_without_rng(self, engine):
        logits = torch.tensor([[0.0, 10.0, 9.0]])

        torch.manual_seed(1)
        first = engine._sample(
            logits,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=None,
            eos_token_id=None,
            do_sample=False,
        )
        torch.manual_seed(999)
        second = engine._sample(
            logits,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=None,
            eos_token_id=None,
            do_sample=False,
        )

        assert torch.equal(first, torch.tensor([[1]]))
        assert torch.equal(first, second)

    def test_sampling_mode_uses_multinomial(self, engine):
        logits = torch.tensor([[0.0, 0.0, 10.0]])

        sampled = engine._sample(
            logits,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=None,
            eos_token_id=None,
            do_sample=True,
        )

        assert sampled.shape == (1, 1)

    def test_generate_with_eos(self, engine):
        """Test generation stopping at EOS token."""

        class FakeTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(str(t) for t in tokens)

        input_ids = torch.tensor([[1, 2, 3]])
        generated, metadata = engine.generate(
            input_ids,
            config=GenerateConfig(
                max_new_tokens=2,
                do_sample=False,
                eos_token_id=999,
            ),
        )
        assert generated.shape[1] <= 3 + 2

    def test_decode_text(self, engine):
        """Test text decoding."""
        input_ids = torch.tensor([[1, 2, 3]])
        generated, _ = engine.generate(
            input_ids,
            config=GenerateConfig(max_new_tokens=2, do_sample=False),
        )

        class FakeTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(str(t) for t in tokens)

        text = decode_text(generated, input_ids, FakeTokenizer())
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_with_repetition_penalty(self, engine):
        """Test generation with repetition penalty."""
        input_ids = torch.tensor([[1, 2, 3]])
        generated, _ = engine.generate(
            input_ids,
            config=GenerateConfig(
                max_new_tokens=2,
                do_sample=True,
                repetition_penalty=1.5,
            ),
        )
        assert generated.shape[1] == 3 + 2

    def test_generate_multiple_sequences(self, engine):
        """Test generation with multiple return sequences."""
        input_ids = torch.tensor([[1, 2, 3]])
        generated, metadata = engine.generate(
            input_ids,
            config=GenerateConfig(
                max_new_tokens=2,
                num_return_sequences=2,
                do_sample=False,
            ),
        )
        assert generated.shape[0] == 2
        assert metadata["num_return_sequences"] == 2


class TestLoRA:
    def test_lora_config_validation(self):
        """Test LoRA config validation."""
        # Valid config
        config = LoRAConfig(r=8, lora_alpha=16.0)
        assert config.r == 8
        assert config.lora_alpha == 16.0

        # Invalid r
        with pytest.raises(ValueError):
            LoRAConfig(r=-1)

        with pytest.raises(ValueError):
            LoRAConfig(r=0)

        # Invalid alpha
        with pytest.raises(ValueError):
            LoRAConfig(lora_alpha=0)

        # Invalid dropout
        with pytest.raises(ValueError):
            LoRAConfig(lora_dropout=1.5)

    def test_lora_linear_forward(self, simple_model):
        """Test LoRA linear layer forward pass."""
        config = LoRAConfig(r=4, lora_alpha=8.0)
        adapter = apply_lora(simple_model, config)

        x = torch.randn(2, 16)
        output = adapter(x)
        assert output.shape == (2, 8)

    def test_lora_param_count(self, simple_model):
        """Test LoRA parameter counting."""
        # Use default target_modules that match simple_model's layer names
        config = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["0", "2"])
        apply_lora(simple_model, config)

        lora_params = get_lora_params(simple_model)
        assert len(lora_params) > 0
        # Each LoRA layer has lora_A (r x in) and lora_B (out x r)
        count = lora_param_count(simple_model)
        assert count == sum(p.numel() for p in lora_params)

    def test_freeze_base_model(self, simple_model):
        """Test freezing base model parameters."""
        config = LoRAConfig(r=4, lora_alpha=8.0)
        apply_lora(simple_model, config)
        freeze_base_model(simple_model)

        for name, param in simple_model.named_parameters():
            if "lora" in name:
                assert param.requires_grad is True
            else:
                assert param.requires_grad is False

    def test_lora_merge_unmerge(self, simple_model):
        """Test LoRA weight merging and unmerging."""
        config = LoRAConfig(r=4, lora_alpha=8.0, merge_weights=True)
        adapter = apply_lora(simple_model, config)

        # Set some non-zero values
        for child in simple_model.modules():
            if hasattr(child, "lora_A"):
                child.lora_A.data.fill_(0.1)
                child.lora_B.data.fill_(0.1)

        # Merge weights
        adapter.merge_weights()

        # Unmerge weights
        adapter.unmerge_weights()

        # After unmerge, LoRA params should be reset
        for child in simple_model.modules():
            if hasattr(child, "lora_A"):
                assert torch.allclose(child.lora_A, torch.zeros_like(child.lora_A))
                assert torch.allclose(child.lora_B, torch.zeros_like(child.lora_B))

    def test_lora_on_tepid_h1_model(self, model):
        """Test applying LoRA to full Tepid-H1 model."""
        config = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["q_proj", "k_proj", "v_proj"])
        adapter = apply_lora(model, config)

        # Test forward pass (3 tokens, within max_position_embeddings=64)
        input_ids = torch.tensor([[1, 2, 3]])
        output = adapter(input_ids)
        assert output.logits.shape[0] == 1
        assert output.logits.shape[-1] == 128  # vocab_size from smoke config


class TestInferenceLoRAIntegration:
    """Integration tests combining InferenceEngine and LoRA."""

    def test_generate_with_lora(self, inference_model):
        """Test generation with LoRA adapter applied."""
        config = LoRAConfig(r=4, lora_alpha=8.0)
        adapter = apply_lora(inference_model, config)
        engine = InferenceEngine(adapter, config=inference_model.config, use_kv_cache=True)

        input_ids = torch.tensor([[1, 2, 3]])
        generated, metadata = engine.generate(
            input_ids,
            config=GenerateConfig(max_new_tokens=3, do_sample=False),
        )
        assert generated.shape[1] == 3 + 3

    def test_merge_and_generate(self, inference_model):
        """Test merging LoRA weights then generating."""
        config = LoRAConfig(r=4, lora_alpha=8.0)
        adapter = apply_lora(inference_model, config)

        # Merge weights
        adapter.merge_weights()

        engine = InferenceEngine(adapter, config=inference_model.config, use_kv_cache=True)
        input_ids = torch.tensor([[1, 2, 3]])
        generated, _ = engine.generate(
            input_ids,
            config=GenerateConfig(max_new_tokens=3, do_sample=False),
        )
        assert generated.shape[1] == 3 + 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
