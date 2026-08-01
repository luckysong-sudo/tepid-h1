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

### Fixed
- Aligned package metadata version with the public CLI/package version.
- Restored flake8 and mypy as blocking CI gates after clearing the current baseline.
- Fixed NF4 quantization validation order and removed an undefined helper call.
- Added project hygiene regression tests for version metadata and tracked local artifacts.
- Added machine-readable M0-M5 stage-gate auditing with CLI and tests.
- Added machine-readable multi-dimensional project completion reporting.
- Added top-level public API snapshot coverage.
- Added CLI command inventory and JSON schema contract coverage.
- Removed accidental local installer/null artifacts from the tracked tree.
- Type annotation for `attn_bias` parameter in `model.py`
- Removed redundant `num_kv_heads` parameter from `TepidH1CausalLM`
- Unified `self` type hints in `layers.py` to use `nn.Module`
- Removed duplicate exports in `data/__init__.py`
- Fixed syntax errors in test files

### Improved
- Enhanced protocol validation in agent system
- Added explicit `__all__` exports across modules
- Improved error handling with custom exception classes
- Added comprehensive test coverage

## [0.1.0] - 2026-07-31

### Initial Release
- Core model architecture (Delta-Net, GQA, MoE)
- Data governance and decontamination
- Agent runtime with policy enforcement
- Evaluation framework
- CLI interface
