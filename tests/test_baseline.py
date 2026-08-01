import unittest

try:
    import torch
except ImportError:
    torch = None

from tepid_h1.config import TepidH1Config
from tepid_h1.modeling.baseline import (
    TransformerBaselineConfig,
    baseline_parameter_estimate,
    comparison_report,
    hybrid_parameter_estimate,
)


class BaselineEstimateTests(unittest.TestCase):
    def test_active_parameter_proxy_is_matched_within_one_ffn_unit(self):
        model = TepidH1Config.reference_28b_a7b()
        baseline = TransformerBaselineConfig.active_parameter_matched(model)
        estimate = baseline_parameter_estimate(baseline)
        maximum_rounding_gap = model.num_layers * 3 * model.hidden_size / 2

        self.assertLessEqual(abs(estimate["active_parameter_gap"]), maximum_rounding_gap)
        self.assertLess(abs(estimate["active_parameter_gap_percent"]), 0.01)

    def test_report_separates_active_and_physical_moe_parameters(self):
        report = comparison_report(TepidH1Config.prototype())

        self.assertEqual(report["matching_basis"], "per-token active-parameter proxy")
        self.assertGreater(
            report["hybrid"]["physical_parameters"],
            report["hybrid"]["active_parameters"],
        )
        self.assertEqual(
            report["baseline"]["physical_parameters"],
            report["baseline"]["active_parameters"],
        )
        self.assertGreater(hybrid_parameter_estimate(TepidH1Config.smoke())["active_parameters"], 0)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BaselineModelTests(unittest.TestCase):
    def test_baseline_forward_and_loss_are_finite(self):
        from tepid_h1.modeling import TransformerBaselineCausalLM
        from tepid_h1.training import causal_lm_train_step

        config = TransformerBaselineConfig.active_parameter_matched(TepidH1Config.smoke())
        model = TransformerBaselineCausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.model.vocab_size, (1, 7))

        output = model(input_ids, labels=input_ids)
        metrics = causal_lm_train_step(model, input_ids, optimizer)

        self.assertEqual(tuple(output.logits.shape), (1, 7, config.model.vocab_size))
        self.assertEqual(output.delta_states, ())
        self.assertEqual(len(output.attention_states), config.model.num_layers)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIs(output.aux_loss, None)
        torch.testing.assert_close(output.loss, output.language_model_loss)
        self.assertGreater(metrics.gradient_norm, 0)

    def test_baseline_chunked_forward_matches_single_pass(self):
        from tepid_h1.modeling import TransformerBaselineCausalLM

        torch.manual_seed(31)
        config = TransformerBaselineConfig.active_parameter_matched(TepidH1Config.smoke())
        model = TransformerBaselineCausalLM(config).eval()
        input_ids = torch.randint(0, config.model.vocab_size, (1, 7))

        with torch.no_grad():
            full = model(input_ids)
            first = model(input_ids[:, :3])
            second = model(input_ids[:, 3:], attention_states=first.attention_states)
        chunked = torch.cat((first.logits, second.logits), dim=1)

        torch.testing.assert_close(chunked, full.logits, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
