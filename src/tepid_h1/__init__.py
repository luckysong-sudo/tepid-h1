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
from .logging_utils import log_training_step, setup_logging
from .lora import LoRAAdapter, LoRAConfig, apply_lora, freeze_base_model, lora_param_count
from .metrics import MetricBucket, TrainingMetrics
from .mixed_precision import MixedPrecisionConfig, MixedPrecisionManager, PrecisionMode
from .modeling import (
    AttentionState,
    GatedDeltaMemoryEager,
    GatedDeltaMemoryReference,
    GlobalSparseAttentionReference,
    GQAAttentionNative,
    GQAAttentionReference,
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
from .quantization import QuantizationConfig

__all__ = [
    "AgentRuntime",
    "AttentionCache",
    "AttentionState",
    "CheckpointedLayer",
    "Conversation",
    "ConversationAgent",
    "ConversationTurn",
    "EarlyStopper",
    "FinalAnswer",
    "GQAAttentionNative",
    "GQAAttentionReference",
    "GatedDeltaMemoryEager",
    "GatedDeltaMemoryReference",
    "GenerateConfig",
    "GlobalSparseAttentionReference",
    "InferenceEngine",
    "LayerSpec",
    "LoRAAdapter",
    "LoRAConfig",
    "LossTracker",
    "Message",
    "MetricBucket",
    "MixedPrecisionConfig",
    "MixedPrecisionManager",
    "ModelExporter",
    "ModelValidationError",
    "PrecisionMode",
    "QuantizationConfig",
    "RoutedMoEReference",
    "RuntimeDependencies",
    "RuntimeState",
    "SequenceMixer",
    "SwiGLU",
    "TepidH1CausalLM",
    "TepidH1Config",
    "TepidH1Model",
    "TepidH1Output",
    "ToolCall",
    "ToolResult",
    "TrainingCallback",
    "TrainingMetrics",
    "TrainingMetricsBuffer",
    "TrainingRunner",
    "TransformerBaselineCausalLM",
    "TransformerBaselineConfig",
    "TransformerBaselineModel",
    "apply_gradient_checkpointing",
    "apply_lora",
    "baseline_parameter_estimate",
    "comparison_report",
    "decode_text",
    "estimate_memory_savings",
    "freeze_base_model",
    "hybrid_parameter_estimate",
    "log_training_step",
    "lora_param_count",
    "setup_logging",
    "wrap_layers_with_checkpointing",
]
__version__ = "0.2.0"

