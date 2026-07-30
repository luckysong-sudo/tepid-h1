from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorpusStats:
    """Summary statistics for a governed paired-corpus JSONL file."""

    file_path: str
    record_count: int
    source_ids: tuple[str, ...]
    domains: tuple[str, ...]
    records_by_source: dict[str, int]
    records_by_domain: dict[str, int]
    total_token_ids: int
    min_sequence_length: int
    max_sequence_length: int
    mean_sequence_length: float
    unique_token_ids: int
    token_id_min: int
    token_id_max: int
    duplicate_record_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_paired_corpus_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a paired-corpus JSONL file and return its raw records.

    Each line must be a JSON object with at least ``id``, ``source_id``,
    ``domain`` and ``token_ids`` fields. This loader is intentionally permissive
    about extra fields so it can be reused for ad-hoc inspection; the governed
    training path in ``experiments.py`` performs the strict audit-bound check.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
            if not isinstance(item, dict):
                raise TypeError(f"line {line_number} of {path} must be a JSON object")
            for field_name in ("id", "source_id", "domain", "token_ids"):
                if field_name not in item:
                    raise ValueError(f"line {line_number} of {path} is missing field {field_name!r}")
            records.append(item)
    return records


def summarize_paired_corpus(path: str | Path) -> CorpusStats:
    """Compute summary statistics for a governed paired-corpus JSONL file."""
    records = load_paired_corpus_records(path)
    if not records:
        raise ValueError(f"corpus {path} must contain at least one record")

    record_ids: list[str] = []
    source_ids: list[str] = []
    domains: list[str] = []
    sequence_lengths: list[int] = []
    token_counter: Counter[int] = Counter()
    token_min: int | None = None
    token_max: int | None = None

    for record in records:
        record_id = str(record["id"])
        source_id = str(record["source_id"])
        domain = str(record["domain"])
        token_ids = record["token_ids"]
        if not isinstance(token_ids, list):
            raise TypeError(f"record {record_id!r} token_ids must be a list")
        record_ids.append(record_id)
        source_ids.append(source_id)
        domains.append(domain)
        sequence_lengths.append(len(token_ids))
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(
                    f"record {record_id!r} contains non-integer token id: {token_id!r}"
                )
            token_counter[token_id] += 1
            if token_min is None or token_id < token_min:
                token_min = token_id
            if token_max is None or token_id > token_max:
                token_max = token_id

    id_counts = Counter(record_ids)
    duplicates = tuple(sorted(id for id, count in id_counts.items() if count > 1))

    return CorpusStats(
        file_path=str(Path(path)),
        record_count=len(records),
        source_ids=tuple(sorted(set(source_ids))),
        domains=tuple(sorted(set(domains))),
        records_by_source=dict(sorted(Counter(source_ids).items())),
        records_by_domain=dict(sorted(Counter(domains).items())),
        total_token_ids=sum(sequence_lengths),
        min_sequence_length=min(sequence_lengths),
        max_sequence_length=max(sequence_lengths),
        mean_sequence_length=sum(sequence_lengths) / len(sequence_lengths),
        unique_token_ids=len(token_counter),
        token_id_min=token_min if token_min is not None else 0,
        token_id_max=token_max if token_max is not None else 0,
        duplicate_record_ids=duplicates,
    )