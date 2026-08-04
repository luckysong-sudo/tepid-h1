#!/usr/bin/env python3
"""
Tepid-H1 Retrieval Evaluation Example

Demonstrates generating and scoring deterministic retrieval benchmarks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.tepid_h1.evaluation import (
    generate_retrieval_suite,
    load_answer_key,
    score_retrieval,
    write_retrieval_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tepid-H1 retrieval evaluation")
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--positions", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--prompts", type=Path, default=Path("retrieval_prompts.jsonl"))
    parser.add_argument("--answers", type=Path, default=Path("retrieval_answers.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("retrieval_predictions.jsonl"))
    return run_evaluation(parser.parse_args())


def run_evaluation(args: argparse.Namespace) -> int:
    """Generate and score retrieval cases."""
    # Generate cases
    cases = generate_retrieval_suite(
        lengths=tuple(args.lengths),
        positions=tuple(args.positions),
        seed=args.seed,
    )
    print(f"Generated {len(cases)} retrieval cases")

    # Write suite
    write_retrieval_suite(cases, prompts_path=args.prompts, answers_path=args.answers)
    print(f"Wrote prompts to {args.prompts}")
    print(f"Wrote answers to {args.answers}")

    # Simulate predictions (perfect for demo)
    predictions = {}
    for case in cases:
        predictions[case.case_id] = case.expected_answer
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w") as f:
        for case in cases:
            f.write(f'{{"case_id": "{case.case_id}", "answer": "{case.expected_answer}"}}\n')

    # Score
    answers = load_answer_key(args.answers)
    result = score_retrieval(answers, predictions)
    print("\nResults:")
    print(f"  Accuracy: {result['accuracy']:.2%}")
    print(f"  Coverage: {result['coverage']:.2%}")
    print(f"  Passed: {result['passed']}")

    for length, metrics in result.get("by_length", {}).items():
        print(f"  Length {length}: accuracy={metrics['accuracy']:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())