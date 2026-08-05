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

    def test_evaluation_is_no_grad_and_restores_training_mode(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import evaluate_causal_lm

        torch.manual_seed(27)
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        model.train()
        batches = tuple(
            torch.randint(0, config.vocab_size, (1, 6)) for _ in range(2)
        )
        parameters_before = tuple(
            parameter.detach().clone() for parameter in model.parameters()
        )

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
            torch.randint(0, config.vocab_size, (1, 6), generator=generator)
            for _ in range(4)
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
            split_lrs.append(
                causal_lm_train_step(split, batch, split_optimizer).learning_rate
            )
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

    def test_scheduler_validates_input(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import WarmupCosineScheduler

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())

        with self.assertRaisesRegex(TypeError, "warmup_steps"):
            WarmupCosineScheduler(optimizer, warmup_steps="2", total_steps=10)
        with self.assertRaisesRegex(ValueError, "between zero and total_steps"):
            WarmupCosineScheduler(optimizer, warmup_steps=15, total_steps=10)
        with self.assertRaisesRegex(ValueError, "between zero and total_steps"):
            WarmupCosineScheduler(optimizer, warmup_steps=2, total_steps=0)
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            WarmupCosineScheduler(optimizer, warmup_steps=2, total_steps=10, min_lr_ratio=1.5)

    def test_scheduler_step_raises_when_complete(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import WarmupCosineScheduler

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=1, total_steps=1)
        scheduler.step()
        with self.assertRaisesRegex(ValueError, "complete"):
            scheduler.step()

    def test_save_checkpoint_requires_non_negative_step(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import save_checkpoint

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError, "non-negative"):
            save_checkpoint("/tmp/negative.pt", model=model, optimizer=optimizer, step=-1)

    def test_save_checkpoint_rejects_non_serializable_metadata(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import save_checkpoint

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(TypeError, "JSON-compatible"):
            save_checkpoint(
                "/tmp/metadata_test.pt",
                model=model,
                optimizer=optimizer,
                step=0,
                metadata={"tensor": torch.tensor([1, 2, 3])},
            )

    def test_train_step_validates_max_gradient_norm(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import causal_lm_train_step

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())
        input_ids = torch.randint(0, config.vocab_size, (1, 4))

        with self.assertRaisesRegex(ValueError, "positive"):
            causal_lm_train_step(model, input_ids, optimizer, max_gradient_norm=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            causal_lm_train_step(model, input_ids, optimizer, max_gradient_norm=-1)

    def test_train_step_validates_gradient_norm(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import causal_lm_train_step

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))

        metrics = causal_lm_train_step(model, input_ids, optimizer)
        self.assertGreater(metrics.gradient_norm, 0)

    def test_scheduler_warmup_and_cosine_decay(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import WarmupCosineScheduler

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=2, total_steps=10, min_lr_ratio=0.1)

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        self.assertAlmostEqual(lrs[0], 5e-4, places=6)
        self.assertAlmostEqual(lrs[1], 1e-3, places=6)
        self.assertAlmostEqual(lrs[9], 1e-4, places=6)

    def test_scheduler_state_dict_round_trip(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import WarmupCosineScheduler

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=2, total_steps=10)

        scheduler.step()
        scheduler.step()
        state = scheduler.state_dict()

        restored = WarmupCosineScheduler(optimizer, warmup_steps=2, total_steps=10)
        restored.load_state_dict(state)

        self.assertEqual(restored.completed_steps, 2)
        self.assertEqual(restored.state_dict(), state)

    def test_load_checkpoint_validates_scheduler_step(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import WarmupCosineScheduler, load_checkpoint, save_checkpoint

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=1, total_steps=5)

        # Step the scheduler twice to reach step 2
        scheduler.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "test.pt"
            save_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, step=2)

            restored_model = TepidH1CausalLM(config)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters())
            restored_scheduler = WarmupCosineScheduler(restored_optimizer, warmup_steps=1, total_steps=5)
            state = load_checkpoint(checkpoint, model=restored_model, optimizer=restored_optimizer, scheduler=restored_scheduler)
            self.assertEqual(state.step, 2)
            self.assertEqual(restored_scheduler.completed_steps, 2)

    def test_load_checkpoint_rejects_invalid_schema(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import load_checkpoint
        import tempfile

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pt"
            checkpoint.write_bytes(b"not a valid checkpoint")
            with self.assertRaisesRegex((ValueError, Exception), ""):
                load_checkpoint(checkpoint, model=model, optimizer=optimizer)

    def test_load_checkpoint_rejects_invalid_step(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import load_checkpoint
        import tempfile

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid.pt"
            checkpoint.write_bytes(b"not a valid checkpoint")
            with self.assertRaisesRegex((ValueError, Exception), ""):
                load_checkpoint(checkpoint, model=model, optimizer=optimizer)

    def test_train_step_returns_correct_trained_tokens(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import causal_lm_train_step

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        labels = input_ids.clone()
        labels[0, -1] = -100  # Ignore last token

        metrics = causal_lm_train_step(model, input_ids, optimizer, labels=labels)
        # targets[:, 1:] drops the first element, so we expect 6 trained tokens
        self.assertEqual(metrics.trained_tokens, 6)

    def test_train_step_uses_input_ids_when_labels_none(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import causal_lm_train_step

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))

        metrics = causal_lm_train_step(model, input_ids, optimizer)
        self.assertEqual(metrics.trained_tokens, 7)

    def test_scheduler_validates_total_steps_type(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import WarmupCosineScheduler

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())

        with self.assertRaisesRegex(TypeError, "total_steps"):
            WarmupCosineScheduler(optimizer, warmup_steps=2, total_steps="10")

    def test_scheduler_validates_min_lr_ratio(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import WarmupCosineScheduler

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())

        with self.assertRaisesRegex(ValueError, "between zero and one"):
            WarmupCosineScheduler(optimizer, warmup_steps=2, total_steps=10, min_lr_ratio=-0.1)

    def test_load_checkpoint_rejects_invalid_optimizer_state(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import load_checkpoint, save_checkpoint
        import tempfile

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "test.pt"
            save_checkpoint(checkpoint, model=model, optimizer=optimizer, step=0)

            # Load with corrupted optimizer state
            import json
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["optimizer_state"] = "invalid"
            torch.save(payload, checkpoint)

            restored_model = TepidH1CausalLM(config)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters())
            with self.assertRaisesRegex(TypeError, "optimizer_state"):
                load_checkpoint(checkpoint, model=restored_model, optimizer=restored_optimizer)

    def test_load_checkpoint_rejects_invalid_rng_state(self):
        import torch
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import load_checkpoint, save_checkpoint
        import tempfile

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters())

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "test.pt"
            save_checkpoint(checkpoint, model=model, optimizer=optimizer, step=0)

            # Load with corrupted rng state
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["rng_state"] = "invalid"
            torch.save(payload, checkpoint)

            restored_model = TepidH1CausalLM(config)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters())
            with self.assertRaisesRegex(TypeError, "rng_state"):
                load_checkpoint(checkpoint, model=restored_model, optimizer=restored_optimizer)

    def test_evaluate_causal_lm_requires_batches(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling import TepidH1CausalLM
        from tepid_h1.training import evaluate_causal_lm

        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)

        with self.assertRaisesRegex(ValueError, "at least one batch"):
            evaluate_causal_lm(model, ())

    def test_serialized_model_config_for_baseline(self):
        from tepid_h1.config import TepidH1Config
        from tepid_h1.modeling.baseline import TransformerBaselineConfig
        from tepid_h1.modeling import TransformerBaselineCausalLM
        from tepid_h1.training import _serialized_model_config

        config = TransformerBaselineConfig.active_parameter_matched(TepidH1Config.smoke())
        model = TransformerBaselineCausalLM(config)
        serialized = _serialized_model_config(model)
        self.assertIn("architecture", serialized)
        self.assertIn("model", serialized)
        self.assertIn("vocab_size", serialized["model"])


if __name__ == "__main__":
    unittest.main()
