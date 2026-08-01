# Tepid-H1 改进路线图

## 项目现状总结

### 架构概览
- **框架类型**: MLIR-based 深度学习训练编译器框架
- **核心组件**: 配置系统、模型层、数据治理、训练循环、Agent运行时
- **测试覆盖**: 18个测试文件，覆盖主要模块
- **代码质量**: 完整类型注解、数据类验证、确定性随机种子

### 已完成的优化
1. ✅ model.py - 移除冗余参数，添加类型注解
2. ✅ layers.py - 重构为类方法，添加类型约束
3. ✅ data/__init__.py - 添加显式导出列表
4. ✅ agent/protocols.py - 增强协议验证
5. ✅ 新增3个测试文件

---

## 后续改进建议

### 优先级 P0 - CI/CD 与自动化

#### 1. GitHub Actions 工作流
创建 `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run tests
        run: pytest tests/ -v --tb=short
      
      - name: Type checking
        run: mypy src/tepid_h1 --ignore-missing-imports
  
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Lint with flake8
        run: flake8 src/ tests/
      
      - name: Format check with ruff
        run: ruff check src/ tests/
```

#### 2. 预提交钩子
创建 `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.0
    hooks:
      - id: mypy
        additional_dependencies: [types-setuptools]
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

---

### 优先级 P1 - 测试补全

#### 3. 新增测试文件

**tests/test_cli.py** - CLI命令测试:
```python
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

class CLITests(unittest.TestCase):
    def test_plan_command(self):
        from tepid_h1.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["plan"])
        self.assertEqual(args.command, "plan")
    
    # 更多CLI测试...
```

**tests/test_data_stats.py** - 数据统计测试:
```python
import unittest
import tempfile
from pathlib import Path

class DataStatsTests(unittest.TestCase):
    def test_corpus_summary(self):
        from tepid_h1.data.stats import summarize_paired_corpus
        # 测试统计计算
```

**tests/test_retrieval.py** - 检索评估测试:
```python
import unittest
import tempfile
from pathlib import Path

class RetrievalTests(unittest.TestCase):
    def test_generate_and_score_suite(self):
        from tepid_h1.evaluation.retrieval import (
            generate_retrieval_suite,
            write_retrieval_suite,
            load_answer_key,
            score_retrieval,
        )
        # 端到端测试
```

---

### 优先级 P2 - 功能增强

#### 4. Agent系统集成

增强 `agent/runtime.py` 的错误恢复机制:
```python
class AgentRuntime:
    def __init__(self, dependencies: RuntimeDependencies):
        self._deps = dependencies
        self._max_retries = 3  # 新增重试机制
    
    def _run_with_retry(self, operation: Callable, *args, **kwargs):
        for attempt in range(self._max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(0.1 * (2 ** attempt))  # 指数退避
```

#### 5. 数据管道优化

在 `data/tokenizer_benchmark.py` 中添加缓存:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_tokenize(text_hash: str, tokenizer_path: str) -> list[int]:
    # 带缓存的tokenize函数
```

---

### 优先级 P3 - 文档完善

#### 6. API参考文档

使用 Sphinx 生成API文档:
```rst
.. automodule:: tepid_h1.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tepid_h1.modeling.model
   :members:
   :undoc-members:
```

#### 7. 示例脚本

创建 `examples/` 目录:
- `train_example.py` - 简单训练示例
- `agent_example.py` - Agent使用示例
- `data_pipeline.py` - 数据处理示例

---

## 立即执行项

### 1. 添加缺失的 `__init__.py` 导出

```python
# src/tepid_h1/data/__init__.py
from .audit import AuditFinding, AuditReport, audit_inventory, load_inventory
from .decontamination import (
    ContaminationMatch,
    DecontaminationReport,
    TextRecord,
    character_ngrams,
    compare_corpora,
    file_sha256,
    normalize_text,
    text_sha256,
)
from .stats import CorpusStats, SplitIsolationReport, summarize_paired_corpus
from .tokenizer_benchmark import (
    BenchmarkSample,
    DomainMetrics,
    load_corpus,
    benchmark_candidate,
    corpus_digest,
    select_candidate,
)

__all__ = [
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
    "character_ngrams",
    "compare_corpora",
    "corpus_digest",
    "file_sha256",
    "load_corpus",
    "load_inventory",
    "load_text_records",
    "normalize_text",
    "select_candidate",
    "summarize_paired_corpus",
    "text_sha256",
]
```

### 2. 运行现有测试

```bash
pytest tests/ -v --tb=short
```

---

## 实施计划

### 本周
- [ ] 创建CI/CD工作流
- [ ] 设置预提交钩子
- [ ] 补充缺失的测试

### 下周
- [ ] 完成Agent系统增强
- [ ] 添加数据管道缓存
- [ ] 编写API文档

### 下月
- [ ] 性能基准测试套件
- [ ] 端到端集成测试
- [ ] 生产部署指南

---

## 指标追踪

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| 代码覆盖率 | ~85% | 95%+ |
| 类型注解完整度 | 90% | 100% |
| 测试执行时间 | <30s | <20s |
| CLI命令数 | 9 | 12 |