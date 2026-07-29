from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_VOCAB_SIZES = {64_000, 80_000, 96_000}
REQUIRED_DOMAINS = {"zh", "en", "code"}


@dataclass(frozen=True)
class BenchmarkSample:
    domain: str
    text: str


@dataclass(frozen=True)
class DomainMetrics:
    samples: int
    utf8_bytes: int
    characters: int
    tokens: int
    bytes_per_token: float
    characters_per_token: float


def load_corpus(path: str | Path) -> tuple[BenchmarkSample, ...]:
    samples: list[BenchmarkSample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on corpus line {line_number}") from error
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise TypeError(f"corpus line {line_number} must contain string field 'text'")
            domain = item.get("domain")
            if domain not in REQUIRED_DOMAINS:
                raise ValueError(
                    f"corpus line {line_number} domain must be one of {sorted(REQUIRED_DOMAINS)}"
                )
            if not item["text"]:
                raise ValueError(f"corpus line {line_number} text must not be empty")
            samples.append(BenchmarkSample(domain=domain, text=item["text"]))
    if not samples:
        raise ValueError("benchmark corpus must contain at least one sample")
    missing = REQUIRED_DOMAINS - {sample.domain for sample in samples}
    if missing:
        raise ValueError(f"benchmark corpus is missing domains: {sorted(missing)}")
    return tuple(samples)


def corpus_digest(samples: Iterable[BenchmarkSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.domain.encode())
        digest.update(b"\0")
        digest.update(sample.text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def benchmark_candidate(
    *,
    name: str,
    vocab_size: int,
    encode: Callable[[str], Sequence[int]],
    samples: Sequence[BenchmarkSample],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if vocab_size not in REQUIRED_VOCAB_SIZES:
        raise ValueError(f"vocab_size must be one of {sorted(REQUIRED_VOCAB_SIZES)}")
    if not samples:
        raise ValueError("samples must not be empty")

    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"samples": 0, "utf8_bytes": 0, "characters": 0, "tokens": 0}
    )
    started = clock()
    for sample in samples:
        token_ids = encode(sample.text)
        if not token_ids:
            raise ValueError(f"tokenizer {name!r} emitted no tokens for a non-empty sample")
        stats = totals[sample.domain]
        stats["samples"] += 1
        stats["utf8_bytes"] += len(sample.text.encode())
        stats["characters"] += len(sample.text)
        stats["tokens"] += len(token_ids)
    elapsed = max(clock() - started, 1e-9)

    domains: dict[str, dict[str, Any]] = {}
    for domain, stats in sorted(totals.items()):
        metrics = DomainMetrics(
            **stats,
            bytes_per_token=stats["utf8_bytes"] / stats["tokens"],
            characters_per_token=stats["characters"] / stats["tokens"],
        )
        domains[domain] = asdict(metrics)
    total_tokens = sum(item["tokens"] for item in totals.values())
    total_bytes = sum(item["utf8_bytes"] for item in totals.values())
    return {
        "name": name,
        "vocab_size": vocab_size,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "utf8_bytes_per_second": total_bytes / elapsed,
        "domains": domains,
    }


def select_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sizes = {item.get("vocab_size") for item in candidates}
    if sizes != REQUIRED_VOCAB_SIZES or len(candidates) != 3:
        raise ValueError("comparison requires exactly one 64K, 80K and 96K candidate")
    for candidate in candidates:
        missing = REQUIRED_DOMAINS - set(candidate.get("domains", {}))
        if missing:
            raise ValueError(
                f"candidate {candidate.get('name')!r} is missing domains: {sorted(missing)}"
            )

    compression_max = {
        domain: max(item["domains"][domain]["bytes_per_token"] for item in candidates)
        for domain in REQUIRED_DOMAINS
    }
    throughput_max = max(item["utf8_bytes_per_second"] for item in candidates)
    ranking: list[dict[str, Any]] = []
    for candidate in candidates:
        compression_score = sum(
            candidate["domains"][domain]["bytes_per_token"] / compression_max[domain]
            for domain in REQUIRED_DOMAINS
        ) / len(REQUIRED_DOMAINS)
        throughput_score = candidate["utf8_bytes_per_second"] / throughput_max
        score = 0.7 * compression_score + 0.3 * throughput_score
        ranking.append(
            {
                "name": candidate["name"],
                "vocab_size": candidate["vocab_size"],
                "score": score,
                "compression_score": compression_score,
                "throughput_score": throughput_score,
            }
        )
    ranking.sort(key=lambda item: (-item["score"], item["vocab_size"]))
    return {
        "selected": ranking[0]["name"],
        "selected_vocab_size": ranking[0]["vocab_size"],
        "weights": {
            "domain_balanced_compression": 0.7,
            "input_utf8_byte_throughput": 0.3,
        },
        "ranking": ranking,
    }
