"""Tests for sparse attention memory estimation and block structure analysis."""

import pytest

from tepid_h1.config import TepidH1Config
from tepid_h1.evaluation import (
    SparseAttentionProfile,
    SparseAttentionReport,
    describe_sparse_block_structure,
    estimate_sparse_attention_memory,
)


class TestEstimateSparseAttentionMemory:
    """Test the sparse attention memory estimation tool."""

    def test_returns_report_with_profiles(self):
        config = TepidH1Config.smoke()
        report = estimate_sparse_attention_memory(config, (64, 128, 256))
        assert isinstance(report, SparseAttentionReport)
        assert report.schema_version == 1
        assert len(report.profiles) == 3
        for profile in report.profiles:
            assert isinstance(profile, SparseAttentionProfile)
            assert profile.sequence_length in (64, 128, 256)
            assert profile.dense_kv_bytes > 0
            assert profile.sparse_kv_bytes > 0
            assert profile.sparse_kv_bytes <= profile.dense_kv_bytes
            assert 0.0 <= profile.memory_reduction_ratio <= 1.0
            assert 0.0 <= profile.sparsity_ratio <= 1.0

    def test_memory_reduction_increases_with_sequence_length(self):
        config = TepidH1Config.smoke()
        report = estimate_sparse_attention_memory(config, (32, 64, 128))
        reductions = [p.memory_reduction_ratio for p in report.profiles]
        assert reductions[0] <= reductions[1] <= reductions[2]

    def test_summary_contains_aggregates(self):
        config = TepidH1Config.smoke()
        report = estimate_sparse_attention_memory(config, (64, 128))
        summary = report.summary
        assert summary["profile_count"] == 2
        assert "min_memory_reduction_ratio" in summary
        assert "max_memory_reduction_ratio" in summary
        assert "mean_memory_reduction_ratio" in summary
        assert "min_sparsity_ratio" in summary
        assert "max_sparsity_ratio" in summary
        assert "mean_sparsity_ratio" in summary
        assert summary["dtype_bytes"] == 2

    def test_config_summary_contains_relevant_fields(self):
        config = TepidH1Config.smoke()
        report = estimate_sparse_attention_memory(config, (64,))
        cs = report.config_summary
        assert cs["local_window"] == config.local_window
        assert cs["global_stride"] == config.global_sparse_stride
        assert cs["num_kv_heads"] == config.num_kv_heads
        assert cs["head_dim"] == config.head_dim

    def test_rejects_empty_sequence_lengths(self):
        config = TepidH1Config.smoke()
        with pytest.raises(ValueError, match="not be empty"):
            estimate_sparse_attention_memory(config, ())

    def test_rejects_non_positive_sequence_length(self):
        config = TepidH1Config.smoke()
        with pytest.raises(ValueError, match="positive"):
            estimate_sparse_attention_memory(config, (0,))

    def test_rejects_invalid_dtype_bytes(self):
        config = TepidH1Config.smoke()
        with pytest.raises(ValueError, match="dtype_bytes"):
            estimate_sparse_attention_memory(config, (64,), dtype_bytes=3)

    def test_dtype_bytes_affects_byte_counts(self):
        config = TepidH1Config.smoke()
        report_bf16 = estimate_sparse_attention_memory(config, (128,), dtype_bytes=2)
        report_fp32 = estimate_sparse_attention_memory(config, (128,), dtype_bytes=4)
        bf16_profile = report_bf16.profiles[0]
        fp32_profile = report_fp32.profiles[0]
        assert fp32_profile.dense_kv_bytes == bf16_profile.dense_kv_bytes * 2
        assert fp32_profile.sparse_kv_bytes == bf16_profile.sparse_kv_bytes * 2
        # Ratio should be the same regardless of dtype
        assert fp32_profile.memory_reduction_ratio == pytest.approx(
            bf16_profile.memory_reduction_ratio
        )

    def test_to_dict_serialization(self):
        config = TepidH1Config.smoke()
        report = estimate_sparse_attention_memory(config, (64,))
        payload = report.to_dict()
        assert payload["schema_version"] == 1
        assert len(payload["profiles"]) == 1
        assert "summary" in payload
        assert "config_summary" in payload


class TestDescribeSparseBlockStructure:
    """Test the sparse block structure description tool."""

    def test_returns_block_contract(self):
        config = TepidH1Config.smoke()
        result = describe_sparse_block_structure(config, 256)
        assert result["schema_version"] == 1
        assert result["sequence_length"] == 256
        assert result["production_kernel_status"] == "not_implemented"
        contract = result["block_contract"]
        assert "compressed_blocks" in contract
        assert "recent_blocks" in contract
        assert "query_selected_blocks" in contract

    def test_compressed_blocks_contain_anchor_positions(self):
        config = TepidH1Config.smoke()
        result = describe_sparse_block_structure(config, 128)
        compressed = result["block_contract"]["compressed_blocks"]
        assert compressed["stride"] == config.global_sparse_stride
        assert compressed["count"] > 0
        assert all(isinstance(p, int) for p in compressed["positions"])
        # Positions should be multiples of stride
        for pos in compressed["positions"]:
            assert pos % config.global_sparse_stride == 0

    def test_recent_blocks_contain_window_range(self):
        config = TepidH1Config.smoke()
        result = describe_sparse_block_structure(config, 128)
        recent = result["block_contract"]["recent_blocks"]
        assert recent["window_size"] == config.local_window
        start, end = recent["range"]
        assert end - start <= config.local_window
        assert end == 128

    def test_query_selected_blocks_contain_example(self):
        config = TepidH1Config.smoke()
        result = describe_sparse_block_structure(config, 64)
        selected = result["block_contract"]["query_selected_blocks"]
        example = selected["example"]
        assert example["query_position"] == 63
        assert "local_range" in example
        assert "global_anchors" in example
        assert example["total_selected"] > 0

    def test_rejects_non_positive_sequence_length(self):
        config = TepidH1Config.smoke()
        with pytest.raises(ValueError, match="positive"):
            describe_sparse_block_structure(config, 0)

    def test_reference_fallback_limit_present(self):
        config = TepidH1Config.smoke()
        result = describe_sparse_block_structure(config, 32)
        assert result["reference_fallback_limit"] == config.global_reference_max_tokens


