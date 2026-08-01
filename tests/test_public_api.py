from __future__ import annotations

import unittest

import tepid_h1
import tepid_h1.agent
import tepid_h1.data
import tepid_h1.evaluation
import tepid_h1.integrations
import tepid_h1.modeling


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

EXPECTED_SUBPACKAGE_APIS = {
    tepid_h1.agent: [
        "AgentAction",
        "AgentRuntime",
        "AllowlistPolicy",
        "BudgetExceeded",
        "EvidenceVerifier",
        "FinalAnswer",
        "ListTelemetry",
        "ModelValidationError",
        "PolicyDecision",
        "RetryExhausted",
        "RuntimeDependencies",
        "RuntimeState",
        "StateContextBuilder",
        "ToolCall",
        "ToolRegistry",
        "ToolResult",
    ],
    tepid_h1.data: [
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
    ],
    tepid_h1.evaluation: [
        "DeltaBackendBenchmarkConfig",
        "DeltaBackendValidationConfig",
        "RetrievalCase",
        "RoutedMoEBenchmarkConfig",
        "benchmark_delta_backend",
        "benchmark_routed_moe",
        "generate_retrieval_suite",
        "load_answer_key",
        "load_predictions",
        "score_retrieval",
        "validate_delta_backend",
        "write_retrieval_suite",
    ],
    tepid_h1.integrations: [
        "LocalGPUPreflightConfig",
        "ZeroGPUJobConfig",
        "build_local_gpu_preflight_report",
        "run_zero_gpu_job",
    ],
    tepid_h1.modeling: [
        "AttentionState",
        "GQAAttentionNative",
        "GQAAttentionReference",
        "GlobalSparseAttentionReference",
        "GatedDeltaMemoryEager",
        "GatedDeltaMemoryReference",
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
    ],
}


class PublicApiSnapshotTests(unittest.TestCase):
    def test_public_api_snapshot_is_stable(self) -> None:
        self.assertEqual(tepid_h1.__all__, EXPECTED_PUBLIC_API)

    def test_public_api_names_are_exported(self) -> None:
        missing = [name for name in tepid_h1.__all__ if not hasattr(tepid_h1, name)]
        self.assertEqual(missing, [])

    def test_public_api_has_no_duplicate_names(self) -> None:
        self.assertEqual(len(tepid_h1.__all__), len(set(tepid_h1.__all__)))

    def test_subpackage_api_snapshots_are_stable(self) -> None:
        for module, expected_api in EXPECTED_SUBPACKAGE_APIS.items():
            with self.subTest(module=module.__name__):
                self.assertEqual(module.__all__, expected_api)

    def test_subpackage_api_names_are_exported(self) -> None:
        for module in EXPECTED_SUBPACKAGE_APIS:
            with self.subTest(module=module.__name__):
                missing = [name for name in module.__all__ if not hasattr(module, name)]
                self.assertEqual(missing, [])

    def test_subpackage_apis_have_no_duplicate_names(self) -> None:
        for module in EXPECTED_SUBPACKAGE_APIS:
            with self.subTest(module=module.__name__):
                self.assertEqual(len(module.__all__), len(set(module.__all__)))


if __name__ == "__main__":
    unittest.main()
