# Tepid-H1

Tepid-H1 的可执行参考框架。当前版本用于 **M0—M2 原型验证**，不宣称具备正式
28B 训练或生产推理能力。

## 当前实现

- 8 层宏块及 48 层参考配置生成器；
- Gated Delta Memory 的逐 Token 正确性参考实现；
- 带流式绝对位置跟踪、RoPE 与无 KV 复制原生路径的局部 GQA 精确注意力；
- 全局注意力的安全参考回退（当前不是生产级稀疏内核）；
- Delta 与注意力 KV 状态续传，支持整模型分块一致性验证；
- Dense SwiGLU 与 Top-K Routed MoE 参考实现；
- Tepid-H1 Backbone 与 Causal LM 装配；
- 自回归损失、有限值检查、梯度裁剪，以及绑定治理语料血缘和训练配方的可恢复
  checkpoint 最小训练闭环；
- 答案隔离、精确长度的 8K/32K 检索生成与分维度评分套件；
- 全 GQA＋Dense SwiGLU 的激活参数匹配 Transformer 对照基线；
- 绑定审计清单的固定小语料、重复试验与不确定性统计的混合/基线配对报告；
- 显式 CPU/CUDA 与精度契约、CUDA 同步计时和峰值显存记录；
- Delta 候选后端的前向、状态、梯度、分块一致性与目标设备加速资格门；
- 有界输入、固定核心版本和治理语料校验的 Hugging Face ZeroGPU Space 包；
- 已验证的 ZeroGPU BF16、报告下载与 Bucket 持久化执行证据；
- 外置 Agent Runtime、Policy、Tool、Verifier 协议与可复用参考实现；
- M0—M5 阶段门配置和标准库测试。
- 可失败关闭的数据资产清单审计、去污染检测、64K/80K/96K Tokenizer 对比与语料统计工具。

## 明确限制

`GatedDeltaMemoryReference` 使用 Python 时间循环，只用于公式、梯度和小模型验证。
`GlobalSparseAttentionReference` 在短序列上回退为全因果注意力，尚未实现 NSA
压缩/选择/滑窗三分支。MoE 也采用逐专家分发参考实现。以上模块在 350M 之前必须
分别替换或接入经过数值对照的 Triton/CUDA 后端。

## 快速检查

无需 PyTorch：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=src python3 -m tepid_h1.cli plan --variant reference
PYTHONPATH=src python3 -m tepid_h1.cli stage-gates
```

安装 PyTorch 后运行模型测试：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

完整质量门在 Hugging Face ZeroGPU Space 内执行，包括 Ruff、56 项配置、数据治理、
训练、检索、后端资格、部署适配与分块一致性测试，并将报告持久化到绑定的 Bucket。
项目宿主机只编辑与同步代码，不执行测试。GitHub Actions 自动部署 Space、刷新并恢复
Dev Mode、自动恢复 containerd/OCI 启动故障、校验运行提交、触发 ZeroGPU 质量门，
并保存远端报告，不在 GitHub Runner 重复执行项目测试。

M0 数据资产审计：

```bash
PYTHONPATH=src python -m tepid_h1.cli data-audit configs/data_inventory.example.json
```

阶段门配置审计：

```bash
PYTHONPATH=src python -m tepid_h1.cli stage-gates configs/stage_gates.json
```

真实 Tokenizer 对比的输入与命令约定见 `docs/M0_DATA_GOVERNANCE.md`。

## 目录

```text
src/tepid_h1/
  config.py             模型配置与宏块计划
  modeling/             正确性参考模型
  agent/                外置 Agent Runtime 协议、执行循环与参考实现
  data/                 数据治理、去污染、Tokenizer 对比与语料统计
configs/
  stage_gates.json      分阶段验收门
docs/
  ARCHITECTURE.md       实现边界与后端替换约定
tests/                  配置、Runtime、模型与数据工具测试
```
