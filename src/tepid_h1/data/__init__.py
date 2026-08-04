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
from .lineage import (
    LicenseCompatibilityReport,
    LineageEntry,
    LineageReport,
    LineageTracker,
    check_license_compatibility,
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
    "LicenseCompatibilityReport",
    "LineageEntry",
    "LineageReport",
    "LineageTracker",
    "SplitIsolationReport",
    "TextRecord",
    "audit_inventory",
    "benchmark_candidate",
    "check_license_compatibility",
    "check_paired_corpus_isolation",
    "character_ngrams",
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
