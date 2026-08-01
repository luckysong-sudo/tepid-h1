# Public API reference

This document lists the stable import surface for the Tepid-H1 executable
prototype. It follows the package `__all__` contracts that are protected by
tests.

Tepid-H1 is a reference and governance framework. These APIs expose
correctness-first model components, evaluation utilities, data-governance
helpers, agent-runtime primitives and operational adapters. They are not a
claim of trained 28B production inference capability.

## `tepid_h1`

Top-level exports are intended for examples, notebooks and lightweight
integration code that should not need to know the internal package layout.

- `AgentRuntime` - runs bounded external-agent turns with policy, tools and verification.
- `AttentionState` - carries local-attention KV state and absolute token position.
- `AttentionCache` - fixed-window attention cache helper.
- `CheckpointedLayer` - wraps layers for activation checkpointing.
- `Conversation` - immutable conversation transcript container.
- `ConversationAgent` - small conversation wrapper around an agent runtime.
- `ConversationTurn` - single conversation exchange record.
- `EarlyStopper` - simple validation-loss early stopping helper.
- `FinalAnswer` - terminal agent action payload.
- `GenerateConfig` - autoregressive generation settings.
- `GQAAttentionNative` - native grouped-query attention module.
- `InferenceEngine` - text generation wrapper for Tepid-H1 causal LM models.
- `LoRAAdapter` - low-rank adaptation module.
- `LoRAConfig` - LoRA configuration.
- `GQAAttentionReference` - correctness reference for grouped-query attention.
- `GlobalSparseAttentionReference` - deterministic sparse attention reference for global slots.
- `GatedDeltaMemoryEager` - eager Delta memory implementation.
- `GatedDeltaMemoryReference` - token-loop Delta memory oracle.
- `LayerSpec` - macro-block layer descriptor.
- `LossTracker` - rolling loss aggregation helper.
- `Message` - chat-style message record.
- `MetricBucket` - in-memory named metric accumulator.
- `MixedPrecisionConfig` - mixed-precision runtime configuration.
- `MixedPrecisionManager` - autocast and gradient-scaling coordinator.
- `ModelExporter` - model export helper.
- `ModelValidationError` - agent model-output validation error.
- `PrecisionMode` - supported precision mode enum.
- `QuantizationConfig` - quantization configuration.
- `ProjectStatusReport` - machine-readable project completion report.
- `RuntimeDependencies` - dependency bundle for `AgentRuntime`.
- `apply_lora` - attaches LoRA adapters to supported modules.
- `decode_text` - decodes token ids through a tokenizer-like object.
- `freeze_base_model` - freezes non-LoRA model parameters.
- `lora_param_count` - counts LoRA trainable parameters.
- `RuntimeState` - agent runtime state object.
- `RoutedMoEReference` - correctness-first routed MoE module.
- `SequenceMixer` - sequence-mixer enum used in layer plans.
- `StageGate` - single M0-M5 gate contract.
- `StageGateReport` - stage-gate audit result.
- `StatusDimension` - single project-status dimension.
- `SwiGLU` - dense SwiGLU feed-forward module.
- `ToolCall` - structured agent tool-call request.
- `ToolResult` - structured agent tool-call result.
- `TepidH1CausalLM` - causal language model wrapper.
- `TepidH1Config` - Tepid-H1 model and macro-pattern configuration.
- `TepidH1Model` - backbone model.
- `TepidH1Output` - model output with logits, loss and recurrent states.
- `TrainingMetrics` - training metric record.
- `TrainingRunner` - minimal supervised training loop.
- `TrainingCallback` - training callback protocol.
- `TrainingMetricsBuffer` - bounded training metric buffer.
- `TransformerBaselineCausalLM` - causal LM baseline wrapper.
- `TransformerBaselineConfig` - Transformer baseline configuration.
- `TransformerBaselineModel` - matched Transformer baseline backbone.
- `apply_gradient_checkpointing` - applies checkpoint wrappers to eligible layers.
- `audit_stage_gates` - validates the configured M0-M5 stage gates.
- `baseline_parameter_estimate` - estimates matched baseline parameter count.
- `build_project_status_report` - builds the completion-status payload.
- `comparison_report` - compares Tepid-H1 and baseline parameter estimates.
- `estimate_memory_savings` - estimates checkpointing memory savings.
- `hybrid_parameter_estimate` - estimates Tepid-H1 hybrid parameter count.
- `load_stage_gates` - loads stage-gate JSON configuration.
- `log_training_step` - writes structured training log events.
- `setup_logging` - configures project logging.
- `wrap_layers_with_checkpointing` - wraps a sequence of layers for checkpointing.

## `tepid_h1.agent`

Agent exports define the external runtime boundary. They keep model outputs,
policy decisions, tool execution and evidence verification separate.

- `AgentAction` - structured action returned by an agent model.
- `AgentRuntime` - bounded tool-use runtime.
- `AllowlistPolicy` - tool allowlist policy implementation.
- `BudgetExceeded` - runtime budget exhaustion error.
- `EvidenceVerifier` - minimal evidence verifier.
- `FinalAnswer` - terminal answer action.
- `ListTelemetry` - list-backed telemetry sink.
- `ModelValidationError` - model-output validation error.
- `PolicyDecision` - allow/deny policy result.
- `RetryExhausted` - retry exhaustion error.
- `RuntimeDependencies` - runtime dependency bundle.
- `RuntimeState` - runtime state record.
- `StateContextBuilder` - builds compact runtime context.
- `ToolCall` - model-requested tool invocation.
- `ToolRegistry` - callable tool registry.
- `ToolResult` - tool result payload.

## `tepid_h1.data`

Data exports support governed fixtures, decontamination, tokenizer comparison
and corpus statistics.

- `AuditFinding` - data inventory audit issue.
- `AuditReport` - data inventory audit result.
- `BenchmarkSample` - tokenizer benchmark sample.
- `ContaminationMatch` - n-gram overlap match.
- `CorpusStats` - paired-corpus summary statistics.
- `DecontaminationReport` - decontamination comparison result.
- `DomainMetrics` - tokenizer metrics by domain.
- `SplitIsolationReport` - paired split isolation result.
- `TextRecord` - normalized text record.
- `audit_inventory` - validates a governed inventory file.
- `benchmark_candidate` - evaluates a tokenizer candidate.
- `check_paired_corpus_isolation` - detects overlap across paired splits.
- `character_ngrams` - builds character n-grams.
- `compare_corpora` - compares training and benchmark corpora for contamination.
- `corpus_digest` - hashes corpus records.
- `load_corpus` - loads tokenizer benchmark corpus records.
- `load_inventory` - loads a data inventory JSON file.
- `load_paired_corpus_records` - loads paired-corpus records.
- `load_text_records` - loads text records from JSONL.
- `normalize_text` - normalizes text before comparison.
- `select_candidate` - selects a tokenizer candidate from metrics.
- `summarize_paired_corpus` - summarizes paired-corpus composition.

## `tepid_h1.evaluation`

Evaluation exports provide deterministic retrieval scoring plus backend
validation and benchmark protocols.

- `DeltaBackendBenchmarkConfig` - Delta benchmark configuration.
- `DeltaBackendValidationConfig` - Delta validation configuration.
- `RetrievalCase` - retrieval evaluation case.
- `RoutedMoEBenchmarkConfig` - MoE benchmark configuration.
- `benchmark_delta_backend` - records Delta backend throughput evidence.
- `benchmark_routed_moe` - records reference MoE routing-load evidence.
- `generate_retrieval_suite` - creates retrieval evaluation cases.
- `load_answer_key` - loads retrieval answer keys.
- `load_predictions` - loads retrieval predictions.
- `score_retrieval` - scores retrieval predictions.
- `validate_delta_backend` - validates Delta backend parity.
- `write_retrieval_suite` - writes generated retrieval cases.

## `tepid_h1.integrations`

Integration exports isolate host, accelerator and service-specific concerns
from core model code.

- `LocalGPUPreflightConfig` - local CUDA preflight configuration.
- `ZeroGPUJobConfig` - Hugging Face ZeroGPU job configuration.
- `build_local_gpu_preflight_report` - reports host GPU and PyTorch CUDA readiness.
- `run_zero_gpu_job` - runs a bounded ZeroGPU paired smoke job.

## `tepid_h1.modeling`

Modeling exports expose correctness-first reference modules and matched
baseline helpers.

- `AttentionState` - local-attention streaming state.
- `GQAAttentionNative` - native grouped-query attention.
- `GQAAttentionReference` - explicit GQA correctness reference.
- `GlobalSparseAttentionReference` - local-window plus global-anchor sparse attention reference.
- `GatedDeltaMemoryEager` - eager Delta memory path.
- `GatedDeltaMemoryReference` - reference Delta oracle.
- `RoutedMoEReference` - reference routed MoE layer.
- `SwiGLU` - dense feed-forward layer.
- `TepidH1CausalLM` - causal LM wrapper.
- `TepidH1Model` - Tepid-H1 backbone.
- `TepidH1Output` - model output dataclass.
- `TransformerBaselineCausalLM` - baseline causal LM wrapper.
- `TransformerBaselineConfig` - baseline configuration.
- `TransformerBaselineModel` - baseline backbone.
- `baseline_parameter_estimate` - estimates baseline parameters.
- `comparison_report` - compares hybrid and baseline parameter estimates.
- `hybrid_parameter_estimate` - estimates hybrid model parameters.
