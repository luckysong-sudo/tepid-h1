"""PyTorch correctness reference modules for Tepid-H1."""

from .baseline import (
    TransformerBaselineCausalLM,
    TransformerBaselineConfig,
    TransformerBaselineModel,
    baseline_parameter_estimate,
    comparison_report,
    hybrid_parameter_estimate,
)
from .layers import AttentionState
from .model import TepidH1CausalLM, TepidH1Model, TepidH1Output

__all__ = [
    "AttentionState",
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
