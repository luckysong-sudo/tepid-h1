"""Data-governance and tokenizer-evaluation primitives."""

from .audit import AuditFinding, AuditReport, audit_inventory, load_inventory
from .decontamination import (
    ContaminationMatch,
    DecontaminationReport,
    TextRecord,
    character_ngrams,
    compare_corpora,
    load_text_records,
    normalize_text,
)
from .stats import (
    CorpusStats,
    SplitIsolationReport,
    check_paired_corpus_isolation,
    load_paired_corpus_records,
    summarize_paired_corpus,
)
from .tokenizer_benchmark import (
    BenchmarkSample,
    DomainMetrics,
    benchmark_candidate,
    corpus_digest,
    load_corpus,
    select_candidate,
)

__all__ = [
    "AuditFinding",
    "AuditReport",
    "BenchmarkSample",
    "ContaminationMatch",
    "CorpusStats",
    "DecontaminationReport",
    "DomainMetrics",
    "SplitIsolationReport",
    "TextRecord",
    "audit_inventory",
    "benchmark_candidate",
    "character_ngrams",
    "check_paired_corpus_isolation",
    "compare_corpora",
    "corpus_digest",
    "load_corpus",
    "load_inventory",
    "load_paired_corpus_records",
    "load_text_records",
    "normalize_text",
    "select_candidate",
    "summarize_paired_corpus",
]
