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

    def test_grouped_moe_matches_dispatch_oracle_forward_and_gradients(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import RoutedMoEReference

        def dispatch_oracle(layer, input_tensor):
            original_shape = input_tensor.shape
            flat = input_tensor.reshape(-1, original_shape[-1])
            probabilities = layer.router(flat).softmax(dim=-1)
            weights, indices = probabilities.topk(layer.top_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)

            routed = torch.zeros_like(flat)
            for expert_index, expert in enumerate(layer.experts):
                token_indices, slots = (indices == expert_index).nonzero(as_tuple=True)
                if token_indices.numel() == 0:
                    continue
                expert_output = expert(flat[token_indices])
                expert_output = expert_output * weights[token_indices, slots].unsqueeze(-1)
                routed.index_add_(0, token_indices, expert_output)
            return (layer.shared_expert(flat) + routed).reshape(original_shape)

        torch.manual_seed(37)
        config = TepidH1Config.smoke()
        oracle = RoutedMoEReference(config)
        grouped = RoutedMoEReference(config)
        grouped.load_state_dict(oracle.state_dict())
        oracle_input = torch.randn(2, 5, config.hidden_size, requires_grad=True)
        grouped_input = oracle_input.detach().clone().requires_grad_(True)

        oracle_output = dispatch_oracle(oracle, oracle_input)
        grouped_output = grouped(grouped_input)
        oracle_output.square().mean().backward()
        grouped_output.square().mean().backward()

        torch.testing.assert_close(grouped_output, oracle_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(grouped_input.grad, oracle_input.grad, rtol=1e-5, atol=1e-6)
        for grouped_parameter, oracle_parameter in zip(
            grouped.parameters(),
            oracle.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                grouped_parameter.grad,
                oracle_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )
        self.assertIsNotNone(grouped.last_router_stats)
        self.assertEqual(
            int(grouped.last_router_stats.expert_counts.sum()),
            oracle_input.shape[0] * oracle_input.shape[1] * config.moe_top_k,
        )
        self.assertIsNotNone(grouped.last_router_aux_loss)
        self.assertTrue(torch.isfinite(grouped.last_router_aux_loss))

    def test_causal_lm_adds_moe_auxiliary_loss_when_weighted(self):
        from dataclasses import replace

        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM

        torch.manual_seed(41)
        config = replace(TepidH1Config.smoke(), moe_router_aux_loss_weight=0.05)
        model = TepidH1CausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 7))

        output = model(input_ids, labels=input_ids)

        self.assertIsNotNone(output.language_model_loss)
        self.assertIsNotNone(output.aux_loss)
        self.assertIsNotNone(output.loss)
        torch.testing.assert_close(
            output.loss,
            output.language_model_loss + config.moe_router_aux_loss_weight * output.aux_loss,
        )
        self.assertGreater(float(output.aux_loss.detach()), 0)

    def test_zero_moe_auxiliary_weight_preserves_language_model_loss(self):
        from dataclasses import replace

        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM

        torch.manual_seed(43)
        config = replace(TepidH1Config.smoke(), moe_router_aux_loss_weight=0.0)
        model = TepidH1CausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 7))

        output = model(input_ids, labels=input_ids)

        self.assertIsNotNone(output.language_model_loss)
        self.assertIsNotNone(output.aux_loss)
        torch.testing.assert_close(output.loss, output.language_model_loss)

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

    def test_model_rejects_missing_or_extra_recurrent_states(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM

        torch.manual_seed(47)
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 7))

        with torch.no_grad():
            output = model(input_ids)

        with self.assertRaisesRegex(ValueError, "delta_states"):
            model(input_ids[:, :2], delta_states=output.delta_states[:-1])
        with self.assertRaisesRegex(ValueError, "delta_states"):
            model(input_ids[:, :2], delta_states=output.delta_states + output.delta_states[:1])
        with self.assertRaisesRegex(ValueError, "attention_states"):
            model(input_ids[:, :2], attention_states=output.attention_states[:-1])
        with self.assertRaisesRegex(ValueError, "attention_states"):
            model(
                input_ids[:, :2],
                attention_states=output.attention_states + output.attention_states[:1],
            )

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
                self.assertTrue(all(state.tokens_seen == offset for state in attention_states))
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
        from tepid_h1.modeling.layers import GQAAttentionReference

        config = TepidH1Config.smoke()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, config.max_position_embeddings + 1, config.hidden_size)

        with self.assertRaisesRegex(RuntimeError, "max_position_embeddings"):
            layer(x)

    def test_global_sparse_attention_mask_keeps_local_window_and_global_anchors(self):
        from dataclasses import replace

        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GlobalSparseAttentionReference

        config = replace(TepidH1Config.smoke(), local_window=4, global_sparse_stride=3)
        layer = GlobalSparseAttentionReference(config).eval()

        mask = layer._attention_mask(
            query_length=2,
            key_length=10,
            past_length=8,
            device=torch.device("cpu"),
        )

        self.assertEqual(torch.where(mask[0])[0].tolist(), [0, 3, 5, 6, 7, 8])
        self.assertEqual(torch.where(mask[1])[0].tolist(), [0, 3, 6, 7, 8, 9])

    def test_global_sparse_attention_chunk_boundary_matches_single_pass(self):
        from dataclasses import replace

        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GlobalSparseAttentionReference

        torch.manual_seed(31)
        config = replace(TepidH1Config.smoke(), local_window=4, global_sparse_stride=3)
        layer = GlobalSparseAttentionReference(config).eval()
        x = torch.randn(1, 10, config.hidden_size)

        full_output, full_state = layer(x)
        first_output, state = layer(x[:, :6])
        second_output, chunked_state = layer(x[:, 6:], state)
        chunked_output = torch.cat((first_output, second_output), dim=1)

        torch.testing.assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked_state.key, full_state.key)
        torch.testing.assert_close(chunked_state.value, full_state.value)
        self.assertEqual(chunked_state.tokens_seen, 10)

    def test_global_sparse_attention_rejects_trimmed_state(self):
        from dataclasses import replace

        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import (
            AttentionState,
            GlobalSparseAttentionReference,
        )

        config = replace(TepidH1Config.smoke(), local_window=4, global_sparse_stride=3)
        layer = GlobalSparseAttentionReference(config).eval()
        x = torch.randn(1, 2, config.hidden_size)
        state = AttentionState(
            key=torch.randn(1, config.num_kv_heads, 3, config.head_dim),
            value=torch.randn(1, config.num_kv_heads, 3, config.head_dim),
            tokens_seen=8,
        )

        with self.assertRaisesRegex(ValueError, "complete cached history"):
            layer(x, state)


if __name__ == "__main__":
    unittest.main()
