"""PyTorch correctness reference modules for Tepid-H1."""

from .baseline import (
    TransformerBaselineCausalLM,
    TransformerBaselineConfig,
    TransformerBaselineModel,
    baseline_parameter_estimate,
    comparison_report,
    hybrid_parameter_estimate,
)
from .layers import (
    AttentionState,
    GatedDeltaMemoryEager,
    GatedDeltaMemoryReference,
    GlobalSparseAttentionReference,
    GQAAttentionNative,
    GQAAttentionReference,
    RoutedMoEReference,
    SwiGLU,
)
from .model import TepidH1CausalLM, TepidH1Model, TepidH1Output

__all__ = [
    "AttentionState",
    "GQAAttentionNative",
    "GQAAttentionReference",
    "GatedDeltaMemoryEager",
    "GatedDeltaMemoryReference",
    "GlobalSparseAttentionReference",
    "RoutedMoEReference",
    "SwiGLU",
    "TepidH1CausalLM",
    "TepidH1Model",
    "TepidH1Output",
    "TransformerBaselineCausalLM",
    "TransformerBaselineConfig",
    "TransformerBaselineModel",
    "baseline_parameter_estimate",
    "comparison_report",
    "hybrid_parameter_estimate",
]
