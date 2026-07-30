# Architecture implementation contract

## Macro block

Each eight-layer macro block is assembled from the immutable plan in
`tepid_h1.config.MACRO_PATTERN`. The 48-layer reference variant therefore contains:

- 30 Delta sequence mixers;
- 12 local exact-attention mixers;
- 6 global sparse-attention slots;
- 36 Dense SwiGLU channel mixers;
- 12 Routed MoE channel mixers.

## Backend boundary

Every experimental operator starts as a shape-correct, differentiable PyTorch reference.
An optimized backend may replace it only when:

1. forward output agrees with the reference within a declared dtype-specific tolerance;
2. input and parameter gradients pass numerical comparison;
3. recurrent decode agrees with chunkwise prefill at chunk boundaries;
4. empty, short, ragged and maximum-shape cases pass;
5. end-to-end throughput improves on the target topology.

`tepid-h1 delta-validate` turns the first, second, third and fifth conditions into a
machine-readable qualification report for Delta candidates. See `docs/DELTA_BACKEND.md`.
The model currently uses `GatedDeltaMemoryEager`, which preserves the reference parameter
and state layout while replacing three per-token `einsum` calls with two batched matrix
reads and one fused erase/write rank-one update. `GatedDeltaMemoryReference` remains the
correctness oracle.

## Delta state convention

The reference stores each head state as `[key_dim, value_dim]`. Decay acts on the key
axis. Erase reads from the decayed state using `erase * key`, removes the corresponding
rank-one association, and then writes `key outer (write * value)`. Any optimized kernel
must document a transpose if it chooses `[value_dim, key_dim]`.

## Global sparse slot

The current class is deliberately named `GlobalSparseAttentionReference`. It is a
full-attention oracle limited to short sequences and must never be used to claim sparse
speedups. The production contract will expose compressed blocks, recent blocks and
query-selected blocks separately so their recall and cost can be measured.

## Streaming state convention

Every Delta layer returns one recurrent matrix state. Every attention layer returns one
`AttentionState` containing projected KV tensors in `[batch, kv_head, token, head_dim]`
orientation plus the absolute number of tokens seen. Query and key projections use
interleaved rotary position encoding with the configured `rotary_theta`; cached keys are
stored after rotation. Tracking the absolute position separately from retained KV length
keeps chunked execution positionally identical even after a local-attention cache is
trimmed. Calls fail closed beyond `max_position_embeddings`.

`TepidH1Output` keeps Delta and attention states as separate ordered tuples so a caller can
feed both back on the next chunk without coupling unlike state types.

Local attention retains at most `local_window - 1` previous KV entries: this is sufficient
for the first token of the next chunk and bounds decode memory. The global reference retains
the complete KV history up to `global_reference_max_tokens`. A production sparse backend
may use a different physical layout, but must preserve full-pass versus chunked output
agreement at declared boundaries.

`GQAAttentionReference` explicitly repeats KV heads and remains the independent grouped-query
correctness oracle. Model and baseline paths use `GQAAttentionNative`, which delegates
grouped-query expansion to PyTorch SDPA through `enable_gqa` and therefore does not
materialize repeated KV tensors. Both classes share projections, RoPE, cache layout and
state-dict structure; tests compare their outputs, streaming state, input gradients and all
parameter gradients.

## Agent boundary

The model emits only `ToolCall` or `FinalAnswer`. Credentials, permissions, execution,
long-term memory and completion verification remain outside model weights. A final answer
is returned only after the Verifier accepts it.

`tepid_h1.agent.defaults` provides reusable reference implementations of every runtime
protocol: `StateContextBuilder`, `AllowlistPolicy`, `ToolRegistry`, `EvidenceVerifier` and
`ListTelemetry`. `AllowlistPolicy` is fail-closed by default (an empty allowlist denies
every call). `ToolRegistry` converts handler exceptions into `ToolResult` with `ok=False`
so a tool failure never crashes the loop. `EvidenceVerifier` binds a final answer to its
supporting successful tool results. `ListTelemetry` records a serializable event trace for
audit. These defaults are optional; callers may still provide custom protocol
implementations.