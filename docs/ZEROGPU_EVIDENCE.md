# ZeroGPU execution evidence

## 2026-07-29 governed BF16 smoke

The deployable Gradio adapter completed its first real ZeroGPU allocation through
`himartoffice/Tepid-H1`.

- Space revision: `6206491eaa3358b4fed777fe72e12029a93aafd0`
- Tepid-H1 core revision: `a39d279438ac5e25f893b4d8a5f16047b6469a88`
- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb
- Compute capability: 12.0
- PyTorch / CUDA: 2.11.0+cu130 / 13.0
- Parameter dtype: BF16
- Workload: one step, one trial, batch size 1, sequence length 8
- Governed corpus SHA-256:
  `882c60467be6da41b53ec15a650d25d5e3f612c2e32da332bb258bb36e76aa0b`
- Tokens trained per model: 7
- Hybrid measured throughput: 23.204 tokens/second
- Baseline measured throughput: 158.621 tokens/second
- Hybrid allocator peak: 18,611,712 bytes
- Baseline allocator peak: 19,319,296 bytes
- Persistent report:
  `/data/reports/tepid-h1-09373db5a4e14e229f8e9148d7873a33.json`
  in `himartoffice/Tepid-H1-storage`

This closes the ZeroGPU allocation, CUDA, BF16, report-download and persistent-Bucket
plumbing smoke. It does not establish model quality or a stable performance ratio. A
single measured step has no variance estimate, the smoke model is tiny, and the hybrid
path still contains Python correctness references. Decision-grade comparison requires
longer measurement windows, repeated trials and optimized target-hardware operators.

## 2026-07-29 repeated BF16 smoke

The same public Space subsequently completed the adapter's maximum bounded workload:
five synchronized training steps across three paired trials. Both models processed the
same preloaded batches and were matched on the per-token active-parameter proxy with a
0.0% gap.

- Workload: five steps, three trials, batch size 1, sequence length 8
- Tokens trained per model: 35 per trial, 105 total
- Batch SHA-256:
  `ac718cb1516251d37de74f5de6f8fa2d6aae9d8ba64e492819598a961a8a07ee`
- Hybrid throughput: 121.989 tokens/second mean
  (95% CI 79.927–164.050)
- Baseline throughput: 337.753 tokens/second mean
  (95% CI 274.567–400.939)
- Baseline/hybrid throughput ratio: 2.845 geometric mean
  (95% CI 2.200–3.680)
- Hybrid loss change: +0.0274 mean (95% CI -0.0096–+0.0645)
- Baseline loss change: -0.0617 mean (95% CI -0.1787–+0.0552)
- Hybrid-minus-baseline loss-change difference: +0.0891 mean
  (95% CI -0.0348–+0.2130)
- Hybrid allocator peak: 19,340,288 bytes
- Baseline allocator peak: 19,321,344 bytes
- Persistent report:
  `/data/reports/tepid-h1-5520abb5bae84d3eab0cbcaae71424fa.json`
  in `himartoffice/Tepid-H1-storage`

The throughput interval identifies an engineering bottleneck in the current tiny,
eager/reference hybrid implementation: the baseline is faster under this workload.
This result prioritizes optimization of the Delta, attention and MoE operator paths
before scaling the training window. It is not evidence against the target architecture,
whose optimized kernels do not exist yet.

The paired loss-change interval crosses zero, so this smoke provides no evidence of a
quality difference. Three tiny trials remain far below a decision-grade model-quality
experiment; their purpose is reproducible execution, variance reporting and early
performance diagnosis.

## 2026-07-29 post-optimization repeated BF16 smoke

GitHub Actions run
[`30451774560`](https://github.com/luckysong-sudo/tepid-h1/actions/runs/30451774560)
deployed core revision `ad65470ff2dae84ca41f49d36997d48939a9587f`,
refreshed Space Dev Mode without manual intervention, passed the 19-check remote
quality gate, and then invoked the bounded performance endpoint. The quality gate
included Ruff, 51 unit tests, data governance and decontamination, checkpoint resume,
retrieval evaluation, Delta parity, a paired CUDA smoke, and the dashboard checks.

The performance run used the same device class, configuration, governed corpus,
batch digest, trial seeds and alternating execution order as the earlier repeated
smoke:

- Workload: five steps, three trials, batch size 1, sequence length 8
- Tokens trained per model: 35 per trial, 105 total
- Batch SHA-256:
  `ac718cb1516251d37de74f5de6f8fa2d6aae9d8ba64e492819598a961a8a07ee`
- Hybrid throughput: 132.737 tokens/second mean
  (95% CI 88.356–177.119)
- Baseline throughput: 285.106 tokens/second mean
  (95% CI 256.383–313.829)
- Baseline/hybrid throughput ratio: 2.216 geometric mean
  (95% CI 1.690–2.905)
- Hybrid loss change: +0.0272 mean (95% CI -0.0100–+0.0644)
- Baseline loss change: -0.0616 mean (95% CI -0.1779–+0.0546)
- Hybrid-minus-baseline loss-change difference: +0.0888 mean
  (95% CI -0.0341–+0.2118)
- Hybrid allocator peak: 19,345,920 bytes
- Baseline allocator peak: 19,326,976 bytes
- Persistent report:
  `/data/reports/tepid-h1-75e6c989e3e74d01b3c7b366eeab49a6.json`
  in `himartoffice/Tepid-H1-storage`

Against the earlier repeated smoke, the hybrid mean throughput increased by 8.8% and
the baseline/hybrid ratio decreased from 2.845 to 2.216, a 22.1% reduction in the
observed relative throughput gap. This is useful directional evidence that the fused
Delta path and native GQA work reduced the bottleneck, but it is not a statistically
isolated optimization effect: the two benchmark runs were not interleaved with each
other, their throughput confidence intervals overlap, and the baseline mean also moved
by -15.6%. The hybrid path therefore remains about 2.2 times slower than the matched
baseline under this tiny workload.

The paired loss-change interval again crosses zero. The architecture changed between
the pre- and post-optimization revisions, so cross-revision loss values are not a
model-quality comparison. Larger target-hardware measurements and optimized
attention, Delta and MoE kernels remain required before making scaling or quality
claims.
