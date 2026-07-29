import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainingTests(unittest.TestCase):
    def test_causal_lm_returns_shifted_loss(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 6))

        output = model(input_ids, labels=input_ids)

        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))
        with self.assertRaisesRegex(ValueError, "same shape"):
            model(input_ids, labels=input_ids[:, :-1])

    def test_train_step_updates_parameters(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import causal_lm_train_step

        torch.manual_seed(23)
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        before = model.model.token_embeddings.weight.detach().clone()

        metrics = causal_lm_train_step(model, input_ids, optimizer)

        self.assertGreater(metrics.loss, 0)
        self.assertGreater(metrics.gradient_norm, 0)
        self.assertEqual(metrics.trained_tokens, 7)
        self.assertFalse(torch.equal(before, model.model.token_embeddings.weight))

    def test_checkpoint_round_trip_restores_model_and_optimizer(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import (
            causal_lm_train_step,
            load_checkpoint,
            save_checkpoint,
        )

        torch.manual_seed(29)
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.vocab_size, (1, 6))
        causal_lm_train_step(model, input_ids, optimizer)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.pt"
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=1,
                metadata={"run": "test"},
            )
            self.assertEqual(list(Path(directory).iterdir()), [checkpoint])
            expected_random = torch.randint(0, 1000, (4,))
            restored = TepidH1CausalLM(config)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            state = load_checkpoint(
                checkpoint,
                model=restored,
                optimizer=restored_optimizer,
            )
            actual_random = torch.randint(0, 1000, (4,))

        self.assertEqual(state.step, 1)
        self.assertEqual(state.metadata, {"run": "test"})
        torch.testing.assert_close(actual_random, expected_random)
        for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(restored_optimizer.state_dict()["state"])

    def test_non_finite_loss_blocks_optimizer_step(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import NonFiniteTrainingError, causal_lm_train_step

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        ignored_labels = torch.full_like(input_ids, -100)
        before = model.model.token_embeddings.weight.detach().clone()

        with self.assertRaisesRegex(NonFiniteTrainingError, "loss"):
            causal_lm_train_step(model, input_ids, optimizer, labels=ignored_labels)

        self.assertTrue(torch.equal(before, model.model.token_embeddings.weight))
        self.assertFalse(optimizer.state_dict()["state"])

    def test_checkpoint_rejects_different_config(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import load_checkpoint, save_checkpoint

        smoke = TepidH1CausalLM(TepidH1Config.smoke())
        optimizer = torch.optim.AdamW(smoke.parameters())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.pt"
            save_checkpoint(checkpoint, model=smoke, optimizer=optimizer, step=0)
            different = TepidH1CausalLM(TepidH1Config.prototype())
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_checkpoint(checkpoint, model=different)

    def test_resume_contract_fails_closed_on_data_or_recipe_change(self):
        from tepid_h1.training import validate_resume_contract

        contract = {
            "schema_version": 1,
            "batch_size": 1,
            "sequence_length": 8,
            "learning_rate": 0.001,
            "seed": 17,
            "data": {
                "kind": "governed_fixed_token_corpus",
                "corpus_file_sha256": "a" * 64,
                "inventory_file_sha256": "b" * 64,
            },
        }
        validate_resume_contract({"training_contract": contract}, contract)

        changed = {**contract, "sequence_length": 16}
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_resume_contract({"training_contract": contract}, changed)
        with self.assertRaisesRegex(ValueError, "does not contain"):
            validate_resume_contract({}, contract)


if __name__ == "__main__":
    unittest.main()
