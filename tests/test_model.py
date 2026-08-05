import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_prototype_forward_shapes(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM

        config = TepidH1Config.prototype()
        model = TepidH1CausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        output = model(input_ids)
        self.assertEqual(tuple(output.logits.shape), (2, 8, config.vocab_size))
        self.assertEqual(len(output.delta_states), 5)
        self.assertEqual(len(output.attention_states), 3)
        self.assertTrue(torch.isfinite(output.logits).all())

        loss = output.logits.float().square().mean()
        loss.backward()
        gradient = model.model.token_embeddings.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_delta_chunk_boundary_matches_single_pass(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GatedDeltaMemoryReference

        torch.manual_seed(7)
        config = TepidH1Config.prototype()
        layer = GatedDeltaMemoryReference(config).eval()
        x = torch.randn(1, 6, config.hidden_size)

        full_output, full_state = layer(x)
        first_output, state = layer(x[:, :3])
        second_output, chunked_state = layer(x[:, 3:], state)
        chunked_output = torch.cat((first_output, second_output), dim=1)

        torch.testing.assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked_state, full_state, rtol=1e-5, atol=1e-6)

    def test_eager_delta_matches_reference_forward_state_and_gradients(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GatedDeltaMemoryEager, GatedDeltaMemoryReference

        torch.manual_seed(9)
        config = TepidH1Config.smoke()
        reference = GatedDeltaMemoryReference(config)
        candidate = GatedDeltaMemoryEager(config)
        candidate.load_state_dict(reference.state_dict())
        reference_input = torch.randn(2, 5, config.hidden_size, requires_grad=True)
        candidate_input = reference_input.detach().clone().requires_grad_(True)
        reference_state = reference.initial_state(
            2,
            device=reference_input.device,
            dtype=reference_input.dtype,
        ).requires_grad_(True)
        candidate_state = reference_state.detach().clone().requires_grad_(True)

        reference_output, reference_final_state = reference(reference_input, reference_state)
        candidate_output, candidate_final_state = candidate(candidate_input, candidate_state)
        (reference_output.square().mean() + reference_final_state.square().mean()).backward()
        (candidate_output.square().mean() + candidate_final_state.square().mean()).backward()

        torch.testing.assert_close(candidate_output, reference_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            candidate_final_state,
            reference_final_state,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(candidate_input.grad, reference_input.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(candidate_state.grad, reference_state.grad, rtol=1e-5, atol=1e-6)
        for candidate_parameter, reference_parameter in zip(
            candidate.parameters(),
            reference.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                candidate_parameter.grad,
                reference_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_model_chunked_forward_matches_single_pass(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM

        torch.manual_seed(11)
        config = TepidH1Config.prototype()
        model = TepidH1CausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 7))

        with torch.no_grad():
            full = model(input_ids)
            first = model(input_ids[:, :3])
            second = model(
                input_ids[:, 3:],
                delta_states=first.delta_states,
                attention_states=first.attention_states,
            )

        chunked_logits = torch.cat((first.logits, second.logits), dim=1)
        torch.testing.assert_close(chunked_logits, full.logits, rtol=1e-5, atol=1e-6)
        self.assertEqual(len(second.delta_states), 5)
        self.assertEqual(len(second.attention_states), 3)

    def test_model_multiple_chunks_match_after_local_cache_trim(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM

        torch.manual_seed(13)
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 33))

        with torch.no_grad():
            full = model(input_ids)
            outputs = []
            delta_states = None
            attention_states = None
            offset = 0
            for width in (5, 13, 7, 8):
                chunk = model(
                    input_ids[:, offset : offset + width],
                    delta_states=delta_states,
                    attention_states=attention_states,
                )
                outputs.append(chunk.logits)
                delta_states = chunk.delta_states
                attention_states = chunk.attention_states
                offset += width
                self.assertTrue(
                    all(state.tokens_seen == offset for state in attention_states)
                )
                self.assertTrue(
                    all(
                        state.key.shape[2] <= config.local_window - 1
                        for state in attention_states[:2]
                    )
                )

        torch.testing.assert_close(
            torch.cat(outputs, dim=1),
            full.logits,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_local_attention_cache_is_bounded(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference

        config = TepidH1Config.prototype()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 6, config.hidden_size)

        _, state = layer(x)
        self.assertEqual(state.key.shape[2], 3)
        self.assertEqual(state.value.shape, state.key.shape)
        self.assertEqual(state.tokens_seen, 6)

    def test_rotary_positions_change_phase_and_preserve_norm(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        projected = torch.randn(1, config.num_query_heads, 1, config.head_dim)

        at_zero = layer._apply_rotary(projected, 0)
        at_one = layer._apply_rotary(projected, 1)

        torch.testing.assert_close(at_zero, projected)
        self.assertFalse(torch.allclose(at_one, projected))
        torch.testing.assert_close(
            at_one.float().norm(dim=-1),
            projected.float().norm(dim=-1),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_native_gqa_matches_explicit_reference_forward_state_and_gradients(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import (
            GQAAttentionNative,
            GQAAttentionReference,
        )

        torch.manual_seed(29)
        config = TepidH1Config.smoke()
        reference = GQAAttentionReference(config, local_window=4)
        candidate = GQAAttentionNative(config, local_window=4)
        candidate.load_state_dict(reference.state_dict())
        reference_input = torch.randn(2, 7, config.hidden_size, requires_grad=True)
        candidate_input = reference_input.detach().clone().requires_grad_(True)

        reference_output, reference_state = reference(reference_input)
        candidate_output, candidate_state = candidate(candidate_input)
        (
            reference_output.square().mean()
            + reference_state.key.square().mean()
            + reference_state.value.square().mean()
        ).backward()
        (
            candidate_output.square().mean()
            + candidate_state.key.square().mean()
            + candidate_state.value.square().mean()
        ).backward()

        torch.testing.assert_close(candidate_output, reference_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(candidate_state.key, reference_state.key)
        torch.testing.assert_close(candidate_state.value, reference_state.value)
        self.assertEqual(candidate_state.tokens_seen, reference_state.tokens_seen)
        torch.testing.assert_close(candidate_input.grad, reference_input.grad, rtol=1e-5, atol=1e-6)
        for candidate_parameter, reference_parameter in zip(
            candidate.parameters(),
            reference.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                candidate_parameter.grad,
                reference_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_local_attention_chunk_boundary_matches_single_pass(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference

        torch.manual_seed(17)
        config = TepidH1Config.prototype()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 7, config.hidden_size)

        full_output, _ = layer(x)
        first_output, state = layer(x[:, :5])
        self.assertEqual(state.key.shape[2], 3)
        self.assertEqual(state.tokens_seen, 5)
        second_output, final_state = layer(x[:, 5:], state)
        chunked_output = torch.cat((first_output, second_output), dim=1)

        torch.testing.assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-6)
        self.assertEqual(final_state.tokens_seen, 7)

    def test_local_attention_multiple_cache_trims_preserve_positions(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference

        torch.manual_seed(23)
        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 13, config.hidden_size)

        full_output, _ = layer(x)
        outputs = []
        state = None
        offset = 0
        for width in (2, 5, 1, 5):
            output, state = layer(x[:, offset : offset + width], state)
            outputs.append(output)
            offset += width
            self.assertEqual(state.tokens_seen, offset)
            self.assertLessEqual(state.key.shape[2], 3)

        torch.testing.assert_close(
            torch.cat(outputs, dim=1),
            full_output,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_attention_rejects_positions_beyond_configured_limit(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference, GlobalSparseAttentionReference, AttentionState

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, config.max_position_embeddings + 1, config.hidden_size)

        with self.assertRaisesRegex(RuntimeError, "max_position_embeddings"):
            layer(x)

        # Test global sparse attention reference fallback
        sparse_layer = GlobalSparseAttentionReference(config)
        x_short = torch.randn(1, 2, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            value=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            tokens_seen=2,
        )
        output, next_state = sparse_layer(x_short, state)
        assert output.shape == x_short.shape

    def test_attention_validates_state_shape(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference, AttentionState

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 2, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 2, 16),  # wrong head_dim
            value=torch.randn(1, config.num_kv_heads, 2, 16),
            tokens_seen=2,
        )

        with self.assertRaisesRegex(ValueError, "head_dim"):
            layer(x, state)

    def test_attention_validates_kv_shapes(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference, AttentionState

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 2, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            value=torch.randn(1, config.num_kv_heads, 2, config.head_dim + 1),  # mismatch
            tokens_seen=2,
        )

        with self.assertRaisesRegex(ValueError, "identical shapes"):
            layer(x, state)

    def test_attention_rejects_invalid_tokens_seen_type(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference, AttentionState

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 2, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            value=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            tokens_seen=True,  # bool, not int
        )

        with self.assertRaisesRegex(TypeError, "tokens_seen"):
            layer(x, state)

    def test_attention_rejects_tokens_seen_smaller_than_cache(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference, AttentionState

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 2, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 5, config.head_dim),
            value=torch.randn(1, config.num_kv_heads, 5, config.head_dim),
            tokens_seen=3,  # smaller than cached tokens
        )

        with self.assertRaisesRegex(ValueError, "cannot be smaller"):
            layer(x, state)

    def test_attention_validates_device_and_dtype(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference, AttentionState

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 2, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 2, config.head_dim, dtype=torch.float64),
            value=torch.randn(1, config.num_kv_heads, 2, config.head_dim, dtype=torch.float64),
            tokens_seen=2,
        )

        with self.assertRaisesRegex(ValueError, "same device and dtype"):
            layer(x, state)

    def test_attention_validates_key_ndim(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference, AttentionState

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 2, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 2),  # 3D instead of 4D
            value=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            tokens_seen=2,
        )

        with self.assertRaisesRegex(ValueError, "must have shape"):
            layer(x, state)

    def test_attention_validates_tokens_seen_type(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference, AttentionState

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 2, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            value=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            tokens_seen="invalid",  # string instead of int
        )

        with self.assertRaisesRegex(TypeError, "tokens_seen"):
            layer(x, state)

    def test_global_sparse_attention_rejects_excessive_tokens(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GlobalSparseAttentionReference, AttentionState

        config = TepidH1Config.smoke()
        layer = GlobalSparseAttentionReference(config)
        x = torch.randn(1, 10, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, config.global_reference_max_tokens - 5, config.head_dim),
            value=torch.randn(1, config.num_kv_heads, config.global_reference_max_tokens - 5, config.head_dim),
            tokens_seen=config.global_reference_max_tokens - 5,
        )

        with self.assertRaisesRegex(RuntimeError, "limited to"):
            layer(x, state)

    def test_delta_layer_rejects_attention_state(self):
        from tepid_h1.config import TepidH1Config, SequenceMixer, ChannelMixer
        from tepid_h1.modeling.model import TepidH1Block
        from tepid_h1.modeling.layers import AttentionState

        config = TepidH1Config.smoke()
        delta_seq = config.layer_plan[0].sequence
        delta_ch = config.layer_plan[0].channel
        block = TepidH1Block(config, sequence=delta_seq, channel=delta_ch)

        x = torch.randn(1, 2, config.hidden_size)
        attention_state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            value=torch.randn(1, config.num_kv_heads, 2, config.head_dim),
            tokens_seen=2,
        )

        # Delta layers should reject attention state
        with self.assertRaisesRegex(ValueError, "Delta layers do not accept attention state"):
            block(x, delta_state=None, attention_state=attention_state)

    def test_attention_layer_rejects_delta_state(self):
        from tepid_h1.config import TepidH1Config, SequenceMixer, ChannelMixer
        from tepid_h1.modeling.model import TepidH1Block

        config = TepidH1Config.smoke()
        # Find an attention layer
        attention_seq = None
        for layer_def in config.layer_plan:
            if layer_def.sequence is not None and layer_def.sequence in (SequenceMixer.LOCAL_ATTENTION, SequenceMixer.GLOBAL_SPARSE_ATTENTION):
                attention_seq = layer_def.sequence
                break

        if attention_seq is not None:
            attention_ch = config.layer_plan[0].channel
            block = TepidH1Block(config, sequence=attention_seq, channel=attention_ch)

            x = torch.randn(1, 2, config.hidden_size)
            delta_state = torch.randn(1, 2, config.hidden_size)

            # Attention layers should reject delta state
            with self.assertRaisesRegex(ValueError, "attention layers do not accept Delta state"):
                block(x, delta_state=delta_state)

    def test_model_moe_report_when_no_moe_layers(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.model import TepidH1Model

        config = TepidH1Config.smoke()
        model = TepidH1Model(config)
        model(torch.randint(0, 128, (1, 4)))

        report = model.moe_balance_report()
        assert report["moe_layers"] > 0  # smoke config has MoE layers

    def test_moe_balance_report_rejects_negative_max_load_cv(self):
        from tepid_h1.modeling.layers import MoERouterStats

        stats = MoERouterStats(
            expert_counts=torch.tensor([10, 10, 10, 10]),
            router_probabilities=torch.empty(0),
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            stats.balance_report(max_load_cv=-0.1)


if __name__ == "__main__":
    unittest.main()
