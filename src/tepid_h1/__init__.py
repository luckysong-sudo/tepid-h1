"""Tepid-H1 reference framework."""

from .agent import (
    AgentRuntime,
    FinalAnswer,
    ModelValidationError,
    RuntimeDependencies,
    RuntimeState,
    ToolCall,
    ToolResult,
)
from .agent.conversation import Conversation, ConversationAgent, ConversationTurn, Message
from .callbacks import (
    EarlyStopper,
    LossTracker,
    TrainingCallback,
    TrainingMetricsBuffer,
    TrainingRunner,
)
from .config import LayerSpec, SequenceMixer, TepidH1Config
from .export import ModelExporter
from .gradient_checkpointing import (
    CheckpointedLayer,
    apply_gradient_checkpointing,
    estimate_memory_savings,
    wrap_layers_with_checkpointing,
)
from .inference import GenerateConfig, InferenceEngine, decode_text
from .lora import LoRAAdapter, LoRAConfig, apply_lora, freeze_base_model, lora_param_count
from .logging_utils import log_training_step, setup_logging
from .metrics import MetricBucket, TrainingMetrics
from .mixed_precision import MixedPrecisionConfig, MixedPrecisionManager, PrecisionMode
from .modeling import (
    AttentionState,
    GQAAttentionNative,
    GQAAttentionReference,
    GlobalSparseAttentionReference,
    GatedDeltaMemoryEager,
    GatedDeltaMemoryReference,
    RoutedMoEReference,
    SwiGLU,
    TepidH1CausalLM,
    TepidH1Model,
    TepidH1Output,
    TransformerBaselineCausalLM,
    TransformerBaselineConfig,
    TransformerBaselineModel,
    baseline_parameter_estimate,
    comparison_report,
    hybrid_parameter_estimate,
)
from .modeling.cache import AttentionCache
from .project_status import ProjectStatusReport, StatusDimension, build_project_status_report
from .quantization import QuantizationConfig
from .training_improvements import (
    TrainingImprovement,
    count_training_improvements,
    filter_training_improvements,
    get_training_improvement_ids,
    list_training_improvements,
)
from .stage_gates import StageGate, StageGateReport, audit_stage_gates, load_stage_gates

__all__ = [
    "AgentRuntime",
    "AttentionState",
    "AttentionCache",
    "CheckpointedLayer",
    "Conversation",
    "ConversationAgent",
    "ConversationTurn",
    "EarlyStopper",
    "FinalAnswer",
    "GenerateConfig",
    "GQAAttentionNative",
    "InferenceEngine",
    "LoRAAdapter",
    "LoRAConfig",
    "GQAAttentionReference",
    "GlobalSparseAttentionReference",
    "GatedDeltaMemoryEager",
    "GatedDeltaMemoryReference",
    "LayerSpec",
    "LossTracker",
    "Message",
    "MetricBucket",
    "MixedPrecisionConfig",
    "MixedPrecisionManager",
    "ModelExporter",
    "ModelValidationError",
    "PrecisionMode",
    "QuantizationConfig",
    "ProjectStatusReport",
    "RuntimeDependencies",
    "apply_lora",
    "decode_text",
    "freeze_base_model",
    "lora_param_count",
    "RuntimeState",
    "RoutedMoEReference",
    "SequenceMixer",
    "StageGate",
    "StageGateReport",
    "StatusDimension",
    "SwiGLU",
    "ToolCall",
    "ToolResult",
    "TepidH1CausalLM",
    "TepidH1Config",
    "TepidH1Model",
    "TepidH1Output",
    "TrainingMetrics",
    "TrainingRunner",
    "TrainingCallback",
    "TrainingMetricsBuffer",
    "TransformerBaselineCausalLM",
    "TransformerBaselineConfig",
    "TransformerBaselineModel",
    "apply_gradient_checkpointing",
    "audit_stage_gates",
    "baseline_parameter_estimate",
    "build_project_status_report",
    "comparison_report",
    "count_training_improvements",
    "estimate_memory_savings",
    "filter_training_improvements",
    "get_training_improvement_ids",
    "hybrid_parameter_estimate",
    "list_training_improvements",
    "load_stage_gates",
    "log_training_step",
    "setup_logging",
    "TrainingImprovement",
    "wrap_layers_with_checkpointing",
]
__version__ = "0.2.0"
