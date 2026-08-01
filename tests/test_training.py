import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_evaluation_is_no_grad_and_restores_training_mode(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import evaluate_causal_lm

        torch.manual_seed(27)
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        model.train()
        batches = tuple(torch.randint(0, config.vocab_size, (1, 6)) for _ in range(2))
        parameters_before = tuple(parameter.detach().clone() for parameter in model.parameters())

        metrics = evaluate_causal_lm(model, batches)

        self.assertGreater(metrics.loss, 0)
        self.assertGreater(metrics.perplexity, 1)
        self.assertEqual(metrics.evaluated_tokens, 10)
        self.assertEqual(metrics.batches, 2)
        self.assertTrue(model.training)
        for before, after in zip(
            parameters_before,
            model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
            self.assertIsNone(after.grad)

    def test_evaluation_uses_masked_labels_for_token_weighting(self):
        from tepid_h1.training import evaluate_causal_lm

        class FakeModel:
            def __init__(self):
                self.training = True
                self.losses = iter((torch.tensor(2.0), torch.tensor(4.0)))
                self.seen_labels = []

            def eval(self):
                self.training = False

            def train(self, mode=True):
                self.training = mode

            def __call__(self, input_ids, labels=None):
                self.seen_labels.append(labels)
                return SimpleNamespace(loss=next(self.losses))

        model = FakeModel()
        batches = (torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7, 8]]))
        labels = (torch.tensor([[1, 2, -100, 4]]), torch.tensor([[5, -100, -100, 8]]))

        metrics = evaluate_causal_lm(model, batches, labels_batches=labels)

        self.assertTrue(model.training)
        self.assertEqual(model.seen_labels, list(labels))
        self.assertEqual(metrics.evaluated_tokens, 3)
        self.assertAlmostEqual(metrics.loss, 8 / 3)

    def test_evaluation_rejects_mismatched_labels_batches(self):
        from tepid_h1.training import evaluate_causal_lm

        class FakeModel:
            training = True

        with self.assertRaisesRegex(ValueError, "labels_batches"):
            evaluate_causal_lm(
                FakeModel(),
                (torch.tensor([[1, 2, 3]]),),
                labels_batches=(torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])),
            )

    def test_evaluation_rejects_batches_without_target_tokens(self):
        from tepid_h1.training import evaluate_causal_lm

        class FakeModel:
            training = True

            def eval(self):
                self.training = False

            def train(self, mode=True):
                self.training = mode

            def __call__(self, input_ids, labels=None):
                return SimpleNamespace(loss=torch.tensor(1.0))

        with self.assertRaisesRegex(ValueError, "target token"):
            evaluate_causal_lm(
                FakeModel(),
                (torch.tensor([[1, 2, 3]]),),
                labels_batches=(torch.full((1, 3), -100),),
            )

    def test_evaluation_rejects_invalid_label_dtype_before_forward(self):
        from tepid_h1.training import evaluate_causal_lm

        class FakeModel:
            def __init__(self):
                self.training = True
                self.called = False

            def eval(self):
                self.training = False

            def train(self, mode=True):
                self.training = mode

            def __call__(self, input_ids, labels=None):
                self.called = True
                return SimpleNamespace(loss=torch.tensor(1.0))

        model = FakeModel()

        with self.assertRaisesRegex(TypeError, "evaluation labels"):
            evaluate_causal_lm(
                model,
                (torch.tensor([[1, 2, 3]]),),
                labels_batches=(torch.tensor([[1.0, 2.0, 3.0]]),),
            )

        self.assertTrue(model.training)
        self.assertFalse(model.called)

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

    def test_train_step_rejects_batches_without_target_tokens(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import causal_lm_train_step

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        ignored_labels = torch.full_like(input_ids, -100)
        before = model.model.token_embeddings.weight.detach().clone()

        with self.assertRaisesRegex(ValueError, "target token"):
            causal_lm_train_step(model, input_ids, optimizer, labels=ignored_labels)

        self.assertTrue(torch.equal(before, model.model.token_embeddings.weight))
        self.assertFalse(optimizer.state_dict()["state"])

    def test_train_step_rejects_empty_targets_before_model_forward(self):
        from tepid_h1.training import causal_lm_train_step

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.called = False

            def forward(self, input_ids, labels=None):
                self.called = True
                return SimpleNamespace(loss=self.weight * 0.0)

        model = FakeModel()
        optimizer = torch.optim.AdamW(model.parameters())

        with self.assertRaisesRegex(ValueError, "target token"):
            causal_lm_train_step(
                model,
                torch.tensor([[1, 2, 3]]),
                optimizer,
                labels=torch.full((1, 3), -100),
            )

        self.assertFalse(model.called)
        self.assertFalse(optimizer.state_dict()["state"])

    def test_train_step_rejects_mismatched_labels_before_model_forward(self):
        from tepid_h1.training import causal_lm_train_step

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.called = False

            def forward(self, input_ids, labels=None):
                self.called = True
                return SimpleNamespace(loss=self.weight * 0.0)

        model = FakeModel()
        optimizer = torch.optim.AdamW(model.parameters())

        with self.assertRaisesRegex(ValueError, "same shape"):
            causal_lm_train_step(
                model,
                torch.tensor([[1, 2, 3]]),
                optimizer,
                labels=torch.tensor([[1, 2]]),
            )

        self.assertFalse(model.called)
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

    def test_checkpoint_save_rejects_invalid_step_type(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import save_checkpoint

        model = TepidH1CausalLM(TepidH1Config.smoke())
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pt"
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                save_checkpoint(checkpoint, model=model, optimizer=optimizer, step=True)

            self.assertFalse(checkpoint.exists())

    def test_checkpoint_save_rejects_scheduler_step_mismatch(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import WarmupCosineScheduler, save_checkpoint

        model = TepidH1CausalLM(TepidH1Config.smoke())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=0, total_steps=4)
        scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mismatch.pt"
            with self.assertRaisesRegex(ValueError, "scheduler step"):
                save_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=0,
                )

            self.assertFalse(checkpoint.exists())

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
        with self.assertRaisesRegex(TypeError, "does not contain"):
            validate_resume_contract({}, contract)

    def test_scheduler_resume_matches_uninterrupted_training(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import (
            WarmupCosineScheduler,
            causal_lm_train_step,
            load_checkpoint,
            save_checkpoint,
        )

        config = TepidH1Config.smoke()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(67)
        batches = tuple(
            torch.randint(0, config.vocab_size, (1, 6), generator=generator) for _ in range(4)
        )

        def build_training_state():
            torch.manual_seed(61)
            model = TepidH1CausalLM(config)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_steps=2,
                total_steps=4,
                min_lr_ratio=0.1,
            )
            return model, optimizer, scheduler

        uninterrupted, full_optimizer, full_scheduler = build_training_state()
        full_lrs = []
        for batch in batches:
            full_lrs.append(
                causal_lm_train_step(uninterrupted, batch, full_optimizer).learning_rate
            )
            full_scheduler.step()

        split, split_optimizer, split_scheduler = build_training_state()
        split_lrs = []
        for batch in batches[:2]:
            split_lrs.append(causal_lm_train_step(split, batch, split_optimizer).learning_rate)
            split_scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "scheduled.pt"
            save_checkpoint(
                checkpoint,
                model=split,
                optimizer=split_optimizer,
                scheduler=split_scheduler,
                step=2,
            )
            resumed, resumed_optimizer, resumed_scheduler = build_training_state()
            state = load_checkpoint(
                checkpoint,
                model=resumed,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
            )
            for batch in batches[2:]:
                split_lrs.append(
                    causal_lm_train_step(
                        resumed,
                        batch,
                        resumed_optimizer,
                    ).learning_rate
                )
                resumed_scheduler.step()

        self.assertEqual(state.step, 2)
        self.assertEqual(full_lrs, [0.0005, 0.001, 0.001, 0.0001])
        self.assertEqual(split_lrs, full_lrs)
        self.assertEqual(resumed_scheduler.state_dict(), full_scheduler.state_dict())
        for expected, actual in zip(
            uninterrupted.parameters(),
            resumed.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
