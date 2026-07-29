from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import TepidH1Config
from .data import (
    audit_inventory,
    benchmark_candidate,
    load_corpus,
    load_inventory,
    select_candidate,
)
from .data.tokenizer_benchmark import corpus_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tepid-h1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print the macro-block layer plan")
    plan.add_argument("--variant", choices=("prototype", "reference"), default="prototype")
    audit = subparsers.add_parser("data-audit", help="audit an M0 data inventory")
    audit.add_argument("inventory", type=Path)
    audit.add_argument("--report", type=Path)
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
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
