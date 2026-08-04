from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import TepidH1Config
from .data import (
    audit_inventory,
    benchmark_candidate,
    check_paired_corpus_isolation,
    compare_corpora,
    load_corpus,
    load_inventory,
    load_text_records,
    select_candidate,
    summarize_paired_corpus,
)
from .data.decontamination import file_sha256
from .data.tokenizer_benchmark import corpus_digest
from .evaluation import (
    generate_retrieval_suite,
    load_answer_key,
    load_predictions,
    score_retrieval,
    write_retrieval_suite,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tepid-h1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # Version command
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    plan = subparsers.add_parser("plan", help="print the macro-block layer plan")
    plan.add_argument("--variant", choices=("prototype", "reference"), default="prototype")
    audit = subparsers.add_parser("data-audit", help="audit an M0 data inventory")
    audit.add_argument("inventory", type=Path)
    audit.add_argument("--report", type=Path)
    decontamination = subparsers.add_parser(
        "decontaminate",
        help="compare training JSONL against a held-out benchmark JSONL",
    )
    decontamination.add_argument("--training", type=Path, required=True)
    decontamination.add_argument("--benchmark", type=Path, required=True)
    decontamination.add_argument("--ngram-size", type=int, default=5)
    decontamination.add_argument("--threshold", type=float, default=0.8)
    decontamination.add_argument("--report", type=Path)
    tokenizer = subparsers.add_parser(
        "tokenizer-benchmark",
        help="compare 64K, 80K and 96K tokenizers on a zh/en/code JSONL corpus",
    )
    tokenizer.add_argument("--corpus", type=Path, required=True)
    tokenizer.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="VOCAB_SIZE=TOKENIZER_JSON",
        help="repeat exactly three times for 64000, 80000 and 96000",
    )
    tokenizer.add_argument("--report", type=Path)
    training = subparsers.add_parser(
        "train-smoke",
        help="run a tiny deterministic causal-LM training loop",
    )
    training.add_argument("--steps", type=int, default=1)
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--sequence-length", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--adam-beta1", type=float, default=0.9)
    training.add_argument("--adam-beta2", type=float, default=0.95)
    training.add_argument("--adam-epsilon", type=float, default=1e-8)
    training.add_argument("--max-gradient-norm", type=float, default=1.0)
    training.add_argument("--warmup-steps", type=int, default=2)
    training.add_argument("--total-steps", type=int, default=10)
    training.add_argument("--min-lr-ratio", type=float, default=0.1)
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--checkpoint", type=Path)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--corpus", type=Path)
    training.add_argument("--validation-corpus", type=Path)
    training.add_argument("--validation-steps", type=int, default=3)
    training.add_argument("--inventory", type=Path)
    training.add_argument("--report", type=Path)
    retrieval_generate = subparsers.add_parser(
        "retrieval-generate",
        help="generate deterministic 8K/32K exact-retrieval cases",
    )
    retrieval_generate.add_argument("--prompts", type=Path, required=True)
    retrieval_generate.add_argument("--answers", type=Path, required=True)
    retrieval_generate.add_argument(
        "--length",
        type=int,
        action="append",
        dest="lengths",
    )
    retrieval_generate.add_argument(
        "--position",
        type=float,
        action="append",
        dest="positions",
    )
    retrieval_generate.add_argument("--seed", type=int, default=41)
    retrieval_score = subparsers.add_parser(
        "retrieval-score",
        help="score exact retrieval predictions against a separate answer key",
    )
    retrieval_score.add_argument("--answers", type=Path, required=True)
    retrieval_score.add_argument("--predictions", type=Path, required=True)
    retrieval_score.add_argument("--minimum-accuracy", type=float, default=1.0)
    retrieval_score.add_argument("--report", type=Path)
    baseline = subparsers.add_parser(
        "baseline-report",
        help="report an active-parameter-matched Transformer baseline",
    )
    baseline.add_argument(
        "--variant",
        choices=("smoke", "prototype", "reference"),
        default="prototype",
    )
    delta_validation = subparsers.add_parser(
        "delta-validate",
        help="qualify a compiled Delta backend against the correctness reference",
    )
    delta_validation.add_argument("--backend", choices=("eager", "inductor"), default="eager")
    delta_validation.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    delta_validation.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    delta_validation.add_argument("--batch-size", type=int, default=1)
    delta_validation.add_argument("--sequence-length", type=int, default=4)
    delta_validation.add_argument("--iterations", type=int, default=3)
    delta_validation.add_argument("--seed", type=int, default=71)
    delta_validation.add_argument("--target-device-label")
    delta_validation.add_argument("--skip-gradients", action="store_true")
    delta_validation.add_argument("--report", type=Path)
    moe_balance = subparsers.add_parser(
        "moe-balance-report",
        help="report smoke-model MoE routing balance",
    )
    moe_balance.add_argument("--batch-size", type=int, default=1)
    moe_balance.add_argument("--sequence-length", type=int, default=8)
    moe_balance.add_argument("--steps", type=int, default=1)
    moe_balance.add_argument("--seed", type=int, default=97)
    moe_balance.add_argument("--max-load-cv", type=float, default=0.25)
    moe_balance.add_argument("--report", type=Path)
    comparison = subparsers.add_parser(
        "compare-smoke",
        help="train hybrid and matched baseline on identical governed or random-token batches",
    )
    comparison.add_argument("--steps", type=int, default=2)
    comparison.add_argument("--trials", type=int, default=1)
    comparison.add_argument("--batch-size", type=int, default=1)
    comparison.add_argument("--sequence-length", type=int, default=8)
    comparison.add_argument("--learning-rate", type=float, default=1e-3)
    comparison.add_argument("--max-gradient-norm", type=float, default=1.0)
    comparison.add_argument("--seed", type=int, default=37)
    comparison.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    comparison.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    comparison.add_argument("--corpus", type=Path)
    comparison.add_argument("--inventory", type=Path)
    comparison.add_argument("--report", type=Path)
    corpus_stats = subparsers.add_parser(
        "corpus-stats",
        help="summarize a governed paired-corpus JSONL file",
    )
    corpus_stats.add_argument("corpus", type=Path)
    corpus_stats.add_argument("--report", type=Path)
    corpus_compare = subparsers.add_parser(
        "corpus-compare",
        help="check isolation between paired training and validation corpora",
    )
    corpus_compare.add_argument("training", type=Path)
    corpus_compare.add_argument("validation", type=Path)
    corpus_compare.add_argument("--report", type=Path)
    return parser


def _write_payload(payload: dict[str, Any], report: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


def _load_tokenizer(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "tokenizer-benchmark requires: pip install -e '.[tokenizer-eval]'"
        ) from error
    return Tokenizer.from_file(str(path))


def _parse_candidate(value: str) -> tuple[int, Path]:
    size, separator, path = value.partition("=")
    if not separator:
        raise ValueError("candidate must use VOCAB_SIZE=TOKENIZER_JSON format")
    try:
        vocab_size = int(size)
    except ValueError as error:
        raise ValueError(f"invalid candidate vocab size: {size!r}") from error
    return vocab_size, Path(path)


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "plan":
        config = (
            TepidH1Config.prototype()
            if args.variant == "prototype"
            else TepidH1Config.reference_28b_a7b()
        )
        payload = {
            "config": config.to_dict(),
            "module_counts": config.module_counts(),
            "layers": [
                {
                    "index": layer.index + 1,
                    "macro_block": layer.macro_block + 1,
                    "sequence": layer.sequence.value,
                    "channel": layer.channel.value,
                }
                for layer in config.layer_plan
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "data-audit":
        report = audit_inventory(load_inventory(args.inventory))
        _write_payload(report.to_dict(), args.report)
        return 0 if report.passed else 2
    if args.command == "decontaminate":
        report = compare_corpora(
            load_text_records(args.training),
            load_text_records(args.benchmark),
            ngram_size=args.ngram_size,
            similarity_threshold=args.threshold,
        )
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "training_file_sha256": file_sha256(args.training),
            "benchmark_file_sha256": file_sha256(args.benchmark),
            **report.to_dict(),
        }
        _write_payload(payload, args.report)
        return 0 if report.clean else 3
    if args.command == "tokenizer-benchmark":
        samples = load_corpus(args.corpus)
        candidates: list[dict[str, Any]] = []
        for value in args.candidate:
            vocab_size, path = _parse_candidate(value)
            tokenizer = _load_tokenizer(path)
            actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
            if actual_vocab_size != vocab_size:
                raise ValueError(
                    f"{path} declares {vocab_size} tokens but contains {actual_vocab_size}"
                )
            candidates.append(
                benchmark_candidate(
                    name=path.stem,
                    vocab_size=vocab_size,
                    encode=lambda text, instance=tokenizer: instance.encode(text).ids,
                    samples=samples,
                )
            )
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus_sha256": corpus_digest(samples),
            "sample_count": len(samples),
            "candidates": candidates,
            "selection": select_candidate(candidates),
        }
        _write_payload(payload, args.report)
        return 0
    if args.command == "train-smoke":
        if args.steps <= 0 or args.batch_size <= 0:
            raise ValueError("steps and batch-size must be positive")
        if not 2 <= args.sequence_length <= 64:
            raise ValueError("sequence-length must be between 2 and 64")
        if args.learning_rate <= 0:
            raise ValueError("learning-rate must be positive")
        if args.weight_decay < 0:
            raise ValueError("weight-decay must be non-negative")
        if not 0 <= args.adam_beta1 < 1 or not 0 <= args.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if args.adam_epsilon <= 0:
            raise ValueError("adam-epsilon must be positive")
        if args.max_gradient_norm <= 0:
            raise ValueError("max-gradient-norm must be positive")
        if args.resume and args.checkpoint is None:
            raise ValueError("--resume requires --checkpoint")
        if (args.corpus is None) != (args.inventory is None):
            raise ValueError("--corpus and --inventory must be provided together")
        if args.validation_steps <= 0:
            raise ValueError("validation-steps must be positive")
        if args.validation_corpus is not None and args.corpus is None:
            raise ValueError("--validation-corpus requires --corpus and --inventory")

        import torch

        from .experiments import (
            PairedExperimentConfig,
            load_governed_corpus,
            validate_governed_split_isolation,
        )
        from .modeling import TepidH1CausalLM
        from .training import (
            WarmupCosineScheduler,
            causal_lm_train_step,
            evaluate_causal_lm,
            load_checkpoint,
            save_checkpoint,
            validate_resume_contract,
        )

        torch.manual_seed(args.seed)
        config = TepidH1Config.smoke()
        model = TepidH1CausalLM(config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=args.total_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        starting_step = 0
        checkpoint_state = None
        if args.resume:
            if not args.checkpoint.exists():
                raise FileNotFoundError(args.checkpoint)
            checkpoint_state = load_checkpoint(
                args.checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            starting_step = checkpoint_state.step
        if starting_step + args.steps > args.total_steps:
            raise ValueError("requested steps exceed the training schedule")

        governed_corpus = (
            load_governed_corpus(
                args.corpus,
                args.inventory,
                PairedExperimentConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                ),
                vocab_size=config.vocab_size,
                start_step=starting_step,
            )
            if args.corpus is not None
            else None
        )
        validation_corpus = (
            load_governed_corpus(
                args.validation_corpus,
                args.inventory,
                PairedExperimentConfig(
                    steps=args.validation_steps,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                ),
                vocab_size=config.vocab_size,
            )
            if args.validation_corpus is not None
            else None
        )
        if governed_corpus is not None and validation_corpus is not None:
            validate_governed_split_isolation(governed_corpus, validation_corpus)
        validation_contract = (
            {
                "kind": "governed_fixed_token_corpus",
                "corpus_file_sha256": validation_corpus.file_sha256,
                "inventory_file_sha256": validation_corpus.inventory_file_sha256,
                "inventory_id": validation_corpus.inventory_id,
                "source_id": validation_corpus.source_id,
                "steps": args.validation_steps,
                "batch_sha256": validation_corpus.batch_sha256,
            }
            if validation_corpus is not None
            else None
        )
        data_contract = (
            {
                "kind": "governed_fixed_token_corpus",
                "corpus_file_sha256": governed_corpus.file_sha256,
                "inventory_file_sha256": governed_corpus.inventory_file_sha256,
                "inventory_id": governed_corpus.inventory_id,
                "source_id": governed_corpus.source_id,
                "validation": validation_contract,
            }
            if governed_corpus is not None
            else {"kind": "deterministic_random_tokens"}
        )
        training_contract = {
            "schema_version": 1,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "betas": [args.adam_beta1, args.adam_beta2],
                "epsilon": args.adam_epsilon,
                "max_gradient_norm": args.max_gradient_norm,
            },
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_steps": args.warmup_steps,
                "total_steps": args.total_steps,
                "min_lr_ratio": args.min_lr_ratio,
            },
            "data": data_contract,
        }
        if checkpoint_state is not None:
            validate_resume_contract(checkpoint_state.metadata, training_contract)

        validation_before = (
            evaluate_causal_lm(model, validation_corpus.batches)
            if validation_corpus is not None
            else None
        )
        metrics = []
        for step_index in range(args.steps):
            input_ids = (
                governed_corpus.batches[step_index]
                if governed_corpus is not None
                else torch.randint(
                    0,
                    config.vocab_size,
                    (args.batch_size, args.sequence_length),
                )
            )
            metrics.append(
                asdict(
                    causal_lm_train_step(
                        model,
                        input_ids,
                        optimizer,
                        max_gradient_norm=args.max_gradient_norm,
                    )
                )
            )
            scheduler.step()
        validation_after = (
            evaluate_causal_lm(model, validation_corpus.batches)
            if validation_corpus is not None
            else None
        )
        final_step = starting_step + args.steps
        if args.checkpoint is not None:
            save_checkpoint(
                args.checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=final_step,
                metadata={
                    "command": "train-smoke",
                    "training_contract": training_contract,
                },
            )
        data_report = {
            **data_contract,
            "start_step": starting_step,
            "end_step": final_step,
            "batch_sha256": (
                governed_corpus.batch_sha256 if governed_corpus is not None else None
            ),
            "records": governed_corpus.records if governed_corpus is not None else None,
            "domains": list(governed_corpus.domains) if governed_corpus is not None else None,
        }
        validation_report = (
            {
                **validation_contract,
                "records": validation_corpus.records,
                "domains": list(validation_corpus.domains),
                "before": asdict(validation_before),
                "after": asdict(validation_after),
                "loss_change": validation_after.loss - validation_before.loss,
                "perplexity_change": (
                    validation_after.perplexity - validation_before.perplexity
                ),
            }
            if validation_corpus is not None
            and validation_contract is not None
            and validation_before is not None
            and validation_after is not None
            else None
        )
        _write_payload(
            {
                "schema_version": 1,
                "config": config.to_dict(),
                "training_contract": training_contract,
                "data": data_report,
                "starting_step": starting_step,
                "final_step": final_step,
                "metrics": metrics,
                "validation": validation_report,
                "scheduler_state": scheduler.state_dict(),
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            },
            args.report,
        )
        return 0
    if args.command == "retrieval-generate":
        cases = generate_retrieval_suite(
            lengths=tuple(args.lengths or (8192, 32768)),
            positions=tuple(args.positions or (0.1, 0.5, 0.9)),
            seed=args.seed,
        )
        write_retrieval_suite(
            cases,
            prompts_path=args.prompts,
            answers_path=args.answers,
        )
        _write_payload(
            {
                "schema_version": 1,
                "cases": len(cases),
                "lengths": sorted({case.target_tokens for case in cases}),
                "positions": sorted({case.insertion_fraction for case in cases}),
                "prompts": str(args.prompts),
                "answers": str(args.answers),
            },
            None,
        )
        return 0
    if args.command == "retrieval-score":
        payload = score_retrieval(
            load_answer_key(args.answers),
            load_predictions(args.predictions),
            minimum_accuracy=args.minimum_accuracy,
        )
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        payload["answer_key_sha256"] = file_sha256(args.answers)
        payload["predictions_sha256"] = file_sha256(args.predictions)
        _write_payload(payload, args.report)
        return 0 if payload["passed"] else 4
    if args.command == "baseline-report":
        from .modeling import comparison_report

        variants = {
            "smoke": TepidH1Config.smoke,
            "prototype": TepidH1Config.prototype,
            "reference": TepidH1Config.reference_28b_a7b,
        }
        _write_payload(comparison_report(variants[args.variant]()), None)
        return 0
    if args.command == "delta-validate":
        from .evaluation.delta_backend import (
            DeltaBackendValidationConfig,
            validate_delta_backend,
        )

        payload = validate_delta_backend(
            DeltaBackendValidationConfig(
                backend=args.backend,
                device=args.device,
                dtype=args.dtype,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                iterations=args.iterations,
                seed=args.seed,
                target_device_label=args.target_device_label,
                verify_gradients=not args.skip_gradients,
            )
        )
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        _write_payload(payload, args.report)
        return 0 if payload["numerical_passed"] else 5
    if args.command == "moe-balance-report":
        if args.batch_size <= 0 or args.sequence_length <= 0 or args.steps <= 0:
            raise ValueError("batch_size, sequence_length and steps must be positive")
        import torch

        from .modeling.layers import MoERouterStats, RoutedMoEReference
        from .modeling.model import TepidH1Model

        config = TepidH1Config.smoke()
        torch.manual_seed(args.seed)
        model = TepidH1Model(config).eval()
        counts = None
        with torch.no_grad():
            for _ in range(args.steps):
                input_ids = torch.randint(
                    config.vocab_size,
                    (args.batch_size, args.sequence_length),
                )
                model(input_ids)
                step_counts = model.moe_router_counts()
                if step_counts is None:
                    raise RuntimeError("smoke model did not produce MoE routing statistics")
                counts = step_counts if counts is None else counts + step_counts
        routing = MoERouterStats(counts, torch.empty(0)).balance_report(max_load_cv=args.max_load_cv)
        routing["moe_layers"] = sum(
            isinstance(layer.channel_mixer, RoutedMoEReference) for layer in model.layers
        )
        routing["observed_layers"] = routing["moe_layers"]
        payload = {
            "schema_version": 1,
            "experiment": "moe_routing_smoke",
            "interpretation": (
                "Local smoke routing evidence; expert-parallel communication and production "
                "batch behavior require a distributed target-hardware run."
            ),
            "config": {
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "steps": args.steps,
                "seed": args.seed,
                "max_load_cv": args.max_load_cv,
            },
            "routing": routing,
        }
        _write_payload(payload, args.report)
        return 0 if payload["routing"]["passed"] else 4
    if args.command == "compare-smoke":
        from .experiments import (
            PairedExperimentConfig,
            load_governed_corpus,
            run_paired_smoke,
        )

        if (args.corpus is None) != (args.inventory is None):
            raise ValueError("--corpus and --inventory must be provided together")
        experiment_config = PairedExperimentConfig(
            steps=args.steps,
            trials=args.trials,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            learning_rate=args.learning_rate,
            max_gradient_norm=args.max_gradient_norm,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
        )
        corpus = (
            load_governed_corpus(
                args.corpus,
                args.inventory,
                experiment_config,
                vocab_size=TepidH1Config.smoke().vocab_size,
            )
            if args.corpus is not None
            else None
        )
        payload = run_paired_smoke(experiment_config, corpus=corpus)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        _write_payload(payload, args.report)
        return 0
    if args.command == "corpus-stats":
        stats = summarize_paired_corpus(args.corpus)
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus_file_sha256": file_sha256(args.corpus),
            **stats.to_dict(),
        }
        _write_payload(payload, args.report)
        return 0 if not stats.duplicate_record_ids else 6
    if args.command == "corpus-compare":
        report = check_paired_corpus_isolation(args.training, args.validation)
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "training_file_sha256": file_sha256(args.training),
            "validation_file_sha256": file_sha256(args.validation),
            **report.to_dict(),
        }
        _write_payload(payload, args.report)
        return 0 if report.clean else 7
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
