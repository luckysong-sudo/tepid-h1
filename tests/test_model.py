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
