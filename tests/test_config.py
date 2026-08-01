import unittest
from dataclasses import replace

from tepid_h1.config import ChannelMixer, SequenceMixer, TepidH1Config


class ConfigTests(unittest.TestCase):
    def test_reference_counts_match_design(self) -> None:
        config = TepidH1Config.reference_28b_a7b()
        self.assertEqual(
            config.module_counts(),
            {
                "delta": 30,
                "local_attention": 12,
                "global_sparse_attention": 6,
                "dense": 36,
                "moe": 12,
            },
        )

    def test_macro_pattern_is_stable(self) -> None:
        plan = TepidH1Config.prototype().layer_plan
        self.assertEqual(plan[0].sequence, SequenceMixer.DELTA)
        self.assertEqual(plan[1].channel, ChannelMixer.MOE)
        self.assertEqual(plan[7].sequence, SequenceMixer.GLOBAL_SPARSE_ATTENTION)

    def test_invalid_head_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            TepidH1Config(
                vocab_size=100,
                hidden_size=255,
                num_layers=8,
                num_query_heads=4,
                num_kv_heads=2,
                head_dim=64,
                local_window=16,
                dense_intermediate_size=128,
                moe_num_experts=4,
                moe_top_k=2,
                moe_expert_intermediate_size=64,
                moe_shared_intermediate_size=128,
                max_position_embeddings=32,
            )

    def test_rotary_configuration_is_validated(self) -> None:
        config = TepidH1Config.smoke()
        with self.assertRaisesRegex(ValueError, "head_dim must be even"):
            replace(config, hidden_size=28, head_dim=7)
        with self.assertRaisesRegex(ValueError, "rotary_theta"):
            replace(config, rotary_theta=0)

    def test_global_sparse_stride_is_validated(self) -> None:
        config = TepidH1Config.smoke()
        with self.assertRaisesRegex(ValueError, "global_sparse_stride"):
            replace(config, global_sparse_stride=0)

    def test_global_reference_token_limit_is_validated(self) -> None:
        config = TepidH1Config.smoke()
        with self.assertRaisesRegex(ValueError, "global_reference_max_tokens"):
            replace(config, global_reference_max_tokens=0)


if __name__ == "__main__":
    unittest.main()
