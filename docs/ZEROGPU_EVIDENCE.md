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
