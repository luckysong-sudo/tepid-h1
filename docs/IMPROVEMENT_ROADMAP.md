# Tepid-H1 改进路线图

## 当前状态

Tepid-H1 当前是可执行的 M0-M2 原型验证框架，不是正式 28B 训练或生产推理系统。
主线能力已经覆盖参考模型、训练闭环、数据治理、检索评估、Delta 后端资格门、
Agent Runtime、ZeroGPU 适配和 CI 验证。

截至本次整理，本地虚拟环境可收集 485 个测试用例；最近一次本地验证结果为：

```text
485 passed, 8 skipped, 21 subtests passed
```

## 已具备能力

- 8 层宏块计划和 48 层参考配置生成器。
- Gated Delta Memory reference/eager 两条路径及前向、状态、梯度一致性测试。
- 原生 GQA attention、RoPE、局部 KV cache 和分块一致性验证。
- GlobalSparseAttentionReference 作为局部窗口 + 全局锚点的确定性稀疏正确性参考路径。
- Dense SwiGLU、Top-K Routed MoE 和 active-parameter matched baseline。
- 训练 step、评估、warmup cosine scheduler、可恢复 checkpoint 和 resume contract。
- 数据资产审计、去污染、paired corpus 统计与训练/验证隔离检查。
- 8K/32K 精确检索生成和评分。
- Agent Runtime 协议、默认 policy/tool/verifier/telemetry 实现和会话封装。
- LoRA、量化、混合精度、梯度 checkpointing、导出和基础推理工具。
- Hugging Face ZeroGPU bundle、远端质量门和持久化报告证据。
- GitHub Actions、pre-commit、Ruff、flake8 和 mypy 阻塞检查。
- 多维度项目完善度报告，可通过 `tepid-h1 project-status` 输出。
- 顶层 public API 快照测试，防止无意破坏导出面。

## 当前限制

- Delta、MoE 和 global sparse attention 仍以 correctness-first 参考实现为主；
  350M 前必须接入或替换为经数值对照的 Triton/CUDA/Inductor 后端。
- Global sparse slot 目前不是生产稀疏内核，不能用于宣称稀疏加速。
- ZeroGPU smoke 只证明可执行性、治理绑定和小规模性能趋势，不证明模型质量。
- 本地 `.venv` 使用 Python 3.14.6，CI 目标矩阵为 Python 3.10/3.11/3.12；
  关键变更仍应以 CI 矩阵为准。
- mypy/flake8 已完成当前基线化，并作为阻塞 CI 门使用。
- 项目版本、CI 依赖和仓库卫生需要持续保持同步，避免文档漂移。

## 下一阶段优先级

### P0: 工程可信度

- 保持 `pyproject.toml`、`__version__`、CLI `--version` 和 changelog 同步。
- 确保 CI 只运行当前已基线化并可达标的质量门。
- 清理本地安装包、临时输出和错误重定向产物，避免进入版本控制。
- 将 README 和路线图定期按测试收集数、CLI 命令和远端质量门结果刷新。
- 持续保持 mypy/flake8 基线为零，避免重新积累类型和静态检查债务。

### P1: 后端资格门

- 扩展 Delta backend validation 的边界形状、dtype 和 CUDA target 报告。
- 为 MoE grouped GEMM 或 fused dispatch 建立 reference parity 测试。
- 将 global sparse contract 拆成压缩块、近邻块和 query-selected 块的可测接口。
- 保留 reference oracle，不用性能路径替代正确性基线。

### P2: 实验可信度 ✅ 已完成

- 增加更长窗口的 paired smoke，输出重复试验、置信区间和固定 batch digest。
- 把训练/验证 split isolation 作为所有 governed experiment 的默认前置检查。
- 将 ZeroGPU 报告和本地/CI 报告统一成稳定 schema，便于横向比较。
- 建立"性能证据"和"质量证据"的分离口径，避免 tiny smoke 被误读。
- 新增 `test_experiments_edge_cases.py`：15 个边界情况测试，覆盖多 trial 可复现性、不同 seed 数据差异、统计聚合、执行顺序交替、多步骤 resume、循环记录 wrapping、最大/最小边界等。
- 新增 `test_agent_policy_extensions.py`：20 个策略扩展测试，覆盖 RateLimitPolicy、CompositePolicy、ToolSchemaValidator、ContentLengthVerifier、ListTelemetry summary。

### P3: API 与用户体验 ✅ 已完成

- 为 CLI 子命令补充端到端示例和最小输入 fixture。
- 为 Agent Runtime 增加更清晰的失败原因和审计事件说明。
- 完善导出、推理、LoRA、量化模块的 README 级使用路径。
- 为长期接口稳定性扩展模块级 API 和 CLI 输出 schema 快照。
- 新增 `ContentLengthVerifier` 和 `CompositePolicy` 等 Agent 策略扩展，增强安全性。
- 新增 `LineageTracker` 和 `LicenseCompatibility` 检查工具，强化数据治理。
- 新增 `SparseAttentionReport` 和 `SparseAnalysis` 工具，提供稀疏注意力内存分析。
- 新增 `TrainingImprovements` 模块，机器可读地记录所有训练改进证据。
- 扩展 `test_public_api.py` 覆盖新增的顶层导出。

## 近期可执行项

- 跑通 CI 等价的本地质量检查：`pytest`、`ruff check`、`flake8`、`mypy`。
- 新增改动必须维持 mypy/flake8 零问题基线。
- 继续扩展 MoE 优化候选 benchmark fixture 和 CUDA target 报告。
- 更新 ZeroGPU evidence，记录最新核心 revision、质量门结果和报告路径。
- 建立目标硬件上的 Delta/MoE 数值对照后，更新 roadmap 中的后端资格门状态。
