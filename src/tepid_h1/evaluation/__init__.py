"""Deterministic evaluation-suite primitives."""

from .delta_backend import (
    DeltaBackendBenchmarkConfig,
    DeltaBackendValidationConfig,
    benchmark_delta_backend,
    validate_delta_backend,
)
from .moe_backend import RoutedMoEBenchmarkConfig, benchmark_routed_moe
from .retrieval import (
    RetrievalCase,
    generate_retrieval_suite,
    load_answer_key,
    load_predictions,
    score_retrieval,
    write_retrieval_suite,
)
from .sparse_analysis import (
    SparseAttentionProfile,
    SparseAttentionReport,
    describe_sparse_block_structure,
    estimate_sparse_attention_memory,
)

__all__ = [
    "DeltaBackendBenchmarkConfig",
    "DeltaBackendValidationConfig",
    "RetrievalCase",
    "RoutedMoEBenchmarkConfig",
    "SparseAttentionProfile",
    "SparseAttentionReport",
    "benchmark_delta_backend",
    "benchmark_routed_moe",
    "describe_sparse_block_structure",
    "estimate_sparse_attention_memory",
    "generate_retrieval_suite",
    "load_answer_key",
    "load_predictions",
    "score_retrieval",
    "validate_delta_backend",
    "write_retrieval_suite",
]
