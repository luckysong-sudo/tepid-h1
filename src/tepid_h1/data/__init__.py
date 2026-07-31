"""Data-governance and tokenizer-evaluation primitives."""

from .audit import AuditFinding, AuditReport, audit_inventory, load_inventory
from .decontamination import (
    ContaminationMatch,
    DecontaminationReport,
    TextRecord,
    compare_corpora,
    load_text_records,
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
    benchmark_candidate,
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
    "SplitIsolationReport",
    "TextRecord",
    "audit_inventory",
    "benchmark_candidate",
    "check_paired_corpus_isolation",
    "compare_corpora",
    "load_corpus",
    "load_inventory",
    "load_paired_corpus_records",
    "load_text_records",
    "select_candidate",
    "summarize_paired_corpus",
]
