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

    def test_local_attention_cache_is_bounded(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference

        config = TepidH1Config.prototype()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 6, config.hidden_size)

        _, state = layer(x)
        self.assertEqual(state.key.shape[2], 3)
        self.assertEqual(state.value.shape, state.key.shape)

    def test_local_attention_chunk_boundary_matches_single_pass(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.layers import GQAAttentionReference

        torch.manual_seed(17)
        config = TepidH1Config.prototype()
        layer = GQAAttentionReference(config, local_window=4).eval()
        x = torch.randn(1, 7, config.hidden_size)

        full_output, _ = layer(x)
        first_output, state = layer(x[:, :3])
        second_output, _ = layer(x[:, 3:], state)
        chunked_output = torch.cat((first_output, second_output), dim=1)

        torch.testing.assert_close(chunked_output, full_output, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
