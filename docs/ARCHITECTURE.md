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

## Agent boundary

The model emits only `ToolCall` or `FinalAnswer`. Credentials, permissions, execution,
long-term memory and completion verification remain outside model weights. A final answer
is returned only after the Verifier accepts it.

