from __future__ import annotations

import unittest

import tepid_h1


EXPECTED_PUBLIC_API = [
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
    "estimate_memory_savings",
    "hybrid_parameter_estimate",
    "load_stage_gates",
    "log_training_step",
    "setup_logging",
    "wrap_layers_with_checkpointing",
]


class PublicApiSnapshotTests(unittest.TestCase):
    def test_public_api_snapshot_is_stable(self) -> None:
        self.assertEqual(tepid_h1.__all__, EXPECTED_PUBLIC_API)

    def test_public_api_names_are_exported(self) -> None:
        missing = [name for name in tepid_h1.__all__ if not hasattr(tepid_h1, name)]
        self.assertEqual(missing, [])

    def test_public_api_has_no_duplicate_names(self) -> None:
        self.assertEqual(len(tepid_h1.__all__), len(set(tepid_h1.__all__)))


if __name__ == "__main__":
    unittest.main()
