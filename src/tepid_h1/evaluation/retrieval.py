from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    target_tokens: int
    insertion_fraction: float
    insertion_index: int
    prompt: str
    expected_answer: str

    def prompt_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "case_id": self.case_id,
            "task": "key_value_exact_retrieval",
            "reference_tokenizer": "whitespace-v1",
            "target_tokens": self.target_tokens,
            "insertion_fraction": self.insertion_fraction,
            "insertion_index": self.insertion_index,
            "prompt": self.prompt,
        }

    def answer_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "case_id": self.case_id,
            "target_tokens": self.target_tokens,
            "insertion_fraction": self.insertion_fraction,
            "answer": self.expected_answer,
        }


def generate_retrieval_suite(
    *,
    lengths: Sequence[int] = (8192, 32768),
    positions: Sequence[float] = (0.1, 0.5, 0.9),
    seed: int = 41,
) -> tuple[RetrievalCase, ...]:
    if not lengths or any(length < 32 for length in lengths):
        raise ValueError("retrieval lengths must contain integers of at least 32")
    if len(set(lengths)) != len(lengths):
        raise ValueError("retrieval lengths must be unique")
    if not positions or any(not 0.0 < position < 1.0 for position in positions):
        raise ValueError("retrieval positions must be in (0, 1)")
    if len(set(positions)) != len(positions):
        raise ValueError("retrieval positions must be unique")

    generator = random.Random(seed)
    cases: list[RetrievalCase] = []
    for target_tokens in lengths:
        for position in positions:
            key = f"key_{generator.getrandbits(48):012x}"
            value = f"value_{generator.getrandbits(64):016x}"
            needle = ("<needle>", key, value, "</needle>")
            query = ("<query>", key, "</query>")
            distractor_count = target_tokens - len(needle) - len(query)
            before_count = round(distractor_count * position)
            distractors = [
                f"ctx_{index:05x}_{generator.getrandbits(24):06x}"
                for index in range(distractor_count)
            ]
            tokens = [
                *distractors[:before_count],
                *needle,
                *distractors[before_count:],
                *query,
            ]
            if len(tokens) != target_tokens:
                raise AssertionError("retrieval generator produced an invalid token count")
            position_label = round(position * 100)
            cases.append(
                RetrievalCase(
                    case_id=f"retrieval-{target_tokens}-p{position_label:02d}-s{seed}",
                    target_tokens=target_tokens,
                    insertion_fraction=position,
                    insertion_index=before_count,
                    prompt=" ".join(tokens),
                    expected_answer=value,
                )
            )
    return tuple(cases)


def write_retrieval_suite(
    cases: Sequence[RetrievalCase],
    *,
    prompts_path: str | Path,
    answers_path: str | Path,
) -> None:
    if not cases:
        raise ValueError("retrieval suite must not be empty")
    _write_jsonl(prompts_path, [case.prompt_record() for case in cases])
    _write_jsonl(answers_path, [case.answer_record() for case in cases])


def load_answer_key(path: str | Path) -> dict[str, dict[str, Any]]:
    records = _load_jsonl(path)
    answers: dict[str, dict[str, Any]] = {}
    for line_number, record in enumerate(records, start=1):
        case_id = record.get("case_id")
        answer = record.get("answer")
        target_tokens = record.get("target_tokens")
        insertion_fraction = record.get("insertion_fraction")
        if not isinstance(case_id, str) or not case_id:
            raise TypeError(f"answer line {line_number} has invalid case_id")
        if case_id in answers:
            raise ValueError(f"duplicate answer case_id {case_id!r}")
        if not isinstance(answer, str) or not answer:
            raise TypeError(f"answer line {line_number} has invalid answer")
        if not isinstance(target_tokens, int) or isinstance(target_tokens, bool):
            raise TypeError(f"answer line {line_number} has invalid target_tokens")
        if not isinstance(insertion_fraction, (int, float)):
            raise TypeError(f"answer line {line_number} has invalid insertion_fraction")
        answers[case_id] = {
            "answer": answer,
            "target_tokens": target_tokens,
            "insertion_fraction": float(insertion_fraction),
        }
    if not answers:
        raise ValueError("answer key must not be empty")
    return answers


def load_predictions(path: str | Path) -> dict[str, str]:
    records = _load_jsonl(path)
    predictions: dict[str, str] = {}
    for line_number, record in enumerate(records, start=1):
        case_id = record.get("case_id")
        answer = record.get("answer")
        if not isinstance(case_id, str) or not case_id:
            raise TypeError(f"prediction line {line_number} has invalid case_id")
        if case_id in predictions:
            raise ValueError(f"duplicate prediction case_id {case_id!r}")
        if not isinstance(answer, str):
            raise TypeError(f"prediction line {line_number} has invalid answer")
        predictions[case_id] = answer
    return predictions


def score_retrieval(
    answers: dict[str, dict[str, Any]],
    predictions: dict[str, str],
    *,
    minimum_accuracy: float = 1.0,
) -> dict[str, Any]:
    if not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError("minimum_accuracy must be in [0, 1]")
    unknown_predictions = sorted(set(predictions) - set(answers))
    if unknown_predictions:
        raise ValueError(f"predictions contain unknown case ids: {unknown_predictions}")

    correct = 0
    by_length: dict[int, list[bool]] = defaultdict(list)
    by_position: dict[str, list[bool]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    for case_id, expected in answers.items():
        predicted = predictions.get(case_id)
        matched = predicted is not None and predicted.strip() == expected["answer"]
        correct += int(matched)
        by_length[expected["target_tokens"]].append(matched)
        bucket = _position_bucket(expected["insertion_fraction"])
        by_position[bucket].append(matched)
        if not matched:
            failures.append(
                {
                    "case_id": case_id,
                    "target_tokens": expected["target_tokens"],
                    "position": bucket,
                    "reason": "missing" if predicted is None else "incorrect",
                }
            )

    total = len(answers)
    submitted = len(predictions)
    coverage = submitted / total
    accuracy = correct / total
    passed = coverage == 1.0 and accuracy >= minimum_accuracy
    return {
        "schema_version": 1,
        "passed": passed,
        "minimum_accuracy": minimum_accuracy,
        "cases": total,
        "submitted": submitted,
        "correct": correct,
        "coverage": coverage,
        "accuracy": accuracy,
        "by_length": {
            str(length): _aggregate(results) for length, results in sorted(by_length.items())
        },
        "by_position": {
            bucket: _aggregate(by_position[bucket]) for bucket in ("early", "middle", "late")
        },
        "failures": failures,
    }


def _position_bucket(value: float) -> str:
    if value <= 1 / 3:
        return "early"
    if value <= 2 / 3:
        return "middle"
    return "late"


def _aggregate(results: Sequence[bool]) -> dict[str, Any]:
    correct = sum(results)
    return {
        "cases": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else None,
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
            if not isinstance(record, dict):
                raise TypeError(f"line {line_number} of {path} must be a JSON object")
            records.append(record)
    return records


def _write_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(f"{json.dumps(record, ensure_ascii=False)}\n" for record in records)
    destination.write_text(rendered, encoding="utf-8")
