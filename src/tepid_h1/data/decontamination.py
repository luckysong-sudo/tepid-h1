from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    text: str


@dataclass(frozen=True)
class ContaminationMatch:
    training_id: str
    benchmark_id: str
    match_type: str
    similarity: float
    normalized_sha256: str


@dataclass(frozen=True)
class DecontaminationReport:
    clean: bool
    training_records: int
    benchmark_records: int
    contaminated_training_records: int
    contamination_rate: float
    exact_matches: int
    near_matches: int
    ngram_size: int
    similarity_threshold: float
    matches: tuple[ContaminationMatch, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matches"] = [asdict(match) for match in self.matches]
        return payload


def load_text_records(path: str | Path) -> tuple[TextRecord, ...]:
    records: list[TextRecord] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise TypeError(f"line {line_number} of {path} must contain string field 'text'")
            if not item["text"].strip():
                raise ValueError(f"line {line_number} of {path} has empty text")
            record_id = item.get("id", f"line:{line_number}")
            if not isinstance(record_id, str) or not record_id.strip():
                raise TypeError(f"line {line_number} of {path} id must be a non-empty string")
            record_id = record_id.strip()
            if record_id in seen_ids:
                raise ValueError(f"duplicate record id {record_id!r} in {path}")
            seen_ids.add(record_id)
            records.append(TextRecord(record_id=record_id, text=item["text"]))
    if not records:
        raise ValueError(f"text corpus {path} must contain at least one record")
    return tuple(records)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def character_ngrams(text: str, size: int) -> frozenset[str]:
    if size <= 0:
        raise ValueError("ngram_size must be positive")
    normalized = normalize_text(text)
    if len(normalized) <= size:
        return frozenset((normalized,))
    return frozenset(
        normalized[index : index + size] for index in range(len(normalized) - size + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def compare_corpora(
    training: tuple[TextRecord, ...],
    benchmarks: tuple[TextRecord, ...],
    *,
    ngram_size: int = 5,
    similarity_threshold: float = 0.8,
) -> DecontaminationReport:
    if not training or not benchmarks:
        raise ValueError("training and benchmark corpora must not be empty")
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")

    benchmark_hashes: dict[str, list[int]] = defaultdict(list)
    benchmark_ngrams: list[frozenset[str]] = []
    inverted_index: dict[str, set[int]] = defaultdict(set)
    for index, record in enumerate(benchmarks):
        benchmark_hashes[text_sha256(record.text)].append(index)
        grams = character_ngrams(record.text, ngram_size)
        benchmark_ngrams.append(grams)
        for gram in grams:
            inverted_index[gram].add(index)

    matches: list[ContaminationMatch] = []
    contaminated_ids: set[str] = set()
    for record in training:
        normalized_hash = text_sha256(record.text)
        exact_indices = benchmark_hashes.get(normalized_hash, ())
        if exact_indices:
            contaminated_ids.add(record.record_id)
            matches.extend(
                ContaminationMatch(
                    training_id=record.record_id,
                    benchmark_id=benchmarks[index].record_id,
                    match_type="exact",
                    similarity=1.0,
                    normalized_sha256=normalized_hash,
                )
                for index in exact_indices
            )
            continue

        grams = character_ngrams(record.text, ngram_size)
        candidate_indices: set[int] = set()
        for gram in grams:
            candidate_indices.update(inverted_index.get(gram, ()))
        for index in sorted(candidate_indices):
            similarity = jaccard(grams, benchmark_ngrams[index])
            if similarity >= similarity_threshold:
                contaminated_ids.add(record.record_id)
                matches.append(
                    ContaminationMatch(
                        training_id=record.record_id,
                        benchmark_id=benchmarks[index].record_id,
                        match_type="near",
                        similarity=similarity,
                        normalized_sha256=normalized_hash,
                    )
                )

    matches.sort(key=lambda item: (item.training_id, item.benchmark_id, item.match_type))
    exact_matches = sum(match.match_type == "exact" for match in matches)
    near_matches = len(matches) - exact_matches
    contaminated = len(contaminated_ids)
    return DecontaminationReport(
        clean=contaminated == 0,
        training_records=len(training),
        benchmark_records=len(benchmarks),
        contaminated_training_records=contaminated,
        contamination_rate=contaminated / len(training),
        exact_matches=exact_matches,
        near_matches=near_matches,
        ngram_size=ngram_size,
        similarity_threshold=similarity_threshold,
        matches=tuple(matches),
    )
