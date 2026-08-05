# Changelog

All notable changes to the Tepid-H1 project will be documented in this file.

## [Unreleased]

### Added
- Agent retry mechanism with exponential backoff
- Data pipeline tokenization cache
- Training metrics collection utilities
- Structured logging support
- CLI examples and demonstration scripts
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality
- Retrieval evaluation suite with validation tests
- Mixed precision training utilities
- LoRA adapter with fan-in/fan-out support
- KV-cache implementation for efficient inference
- Comprehensive test coverage for quantization, training, and cache modules

### Fixed
- Type annotation for `attn_bias` parameter in `model.py`
- Removed redundant `num_kv_heads` parameter from `TepidH1CausalLM`
- Unified `self` type hints in `layers.py` to use `nn.Module`
- Removed duplicate exports in `data/__init__.py`
- Fixed syntax errors in test files
- Fixed retrieval evaluation test assertions

### Improved
- Enhanced protocol validation in agent system
- Added explicit `__all__` exports across modules
- Improved error handling with custom exception classes
- Added comprehensive test coverage
- Increased test coverage from 87% to 89%
- Added validation tests for attention state shape and type checking
- Added scheduler state validation in training checkpoint load

## [0.1.0] - 2026-07-31

### Initial Release
- Core model architecture (Delta-Net, GQA, MoE)
- Data governance and decontamination
- Agent runtime with policy enforcement
- Evaluation framework
- CLI interface