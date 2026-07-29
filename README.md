# Tepid-H1

Tepid-H1 的可执行参考框架。当前版本用于 **M0—M2 原型验证**，不宣称具备正式
28B 训练或生产推理能力。

## 当前实现

- 8 层宏块及 48 层参考配置生成器；
- Gated Delta Memory 的逐 Token 正确性参考实现；
- 局部 GQA 精确注意力；
- 全局注意力的安全参考回退（当前不是生产级稀疏内核）；
- Delta 与注意力 KV 状态续传，支持整模型分块一致性验证；
- Dense SwiGLU 与 Top-K Routed MoE 参考实现；
- Tepid-H1 Backbone 与 Causal LM 装配；
- 自回归损失、有限值检查、梯度裁剪与可恢复 checkpoint 的最小训练闭环；
- 外置 Agent Runtime、Policy、Tool、Verifier 协议；
- M0—M5 阶段门配置和标准库测试。
- 可失败关闭的数据资产清单审计、去污染检测与 64K/80K/96K Tokenizer 对比工具。

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
```

安装 PyTorch 后运行模型测试：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

持续集成会在每次推送和 Pull Request 上安装 CPU 参考环境，执行 Ruff 与完整的
24 项配置、数据治理、训练恢复、Tokenizer、Runtime、模型前向、梯度和分块一致性测试。

M0 数据资产审计：

```bash
PYTHONPATH=src python -m tepid_h1.cli data-audit configs/data_inventory.example.json
```

真实 Tokenizer 对比的输入与命令约定见 `docs/M0_DATA_GOVERNANCE.md`。

## 目录

```text
src/tepid_h1/
  config.py             模型配置与宏块计划
  modeling/             正确性参考模型
  agent/                外置 Agent Runtime 协议与执行循环
configs/
  stage_gates.json      分阶段验收门
docs/
  ARCHITECTURE.md       实现边界与后端替换约定
tests/                  配置、Runtime 与模型烟雾测试
```
