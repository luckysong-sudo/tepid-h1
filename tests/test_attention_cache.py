import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AttentionCacheTests(unittest.TestCase):
    def test_state_dict_round_trip_restores_cache(self):
        from tepid_h1.modeling.cache import AttentionCache

        cache = AttentionCache()
        key = torch.randn(1, 2, 3)
        value = torch.randn(1, 2, 3)
        cache.update(key, value)

        restored = AttentionCache()
        restored.load_state_dict(cache.state_dict)

        self.assertEqual(restored.seq_len, 2)
        self.assertEqual(restored.device, torch.device("cpu"))
        self.assertEqual(restored.dtype, torch.float32)
        torch.testing.assert_close(restored.k_cache, key)
        torch.testing.assert_close(restored.v_cache, value)

    def test_update_rejects_non_3d_updates_without_mutation(self):
        from tepid_h1.modeling.cache import AttentionCache

        cache = AttentionCache()

        with self.assertRaisesRegex(ValueError, "3D"):
            cache.update(torch.randn(2), torch.randn(2))

        self.assertTrue(cache.is_empty)
        self.assertEqual(cache.seq_len, 0)

    def test_update_rejects_bad_update_before_mutating_existing_cache(self):
        from tepid_h1.modeling.cache import AttentionCache

        cache = AttentionCache()
        key = torch.randn(1, 2, 3)
        value = torch.randn(1, 2, 3)
        cache.update(key, value)

        with self.assertRaisesRegex(ValueError, "3D"):
            cache.update(torch.randn(1, 3), torch.randn(1, 3))

        self.assertEqual(cache.seq_len, 2)
        torch.testing.assert_close(cache.k_cache, key)
        torch.testing.assert_close(cache.v_cache, value)

    def test_load_state_rejects_unpaired_cache_tensors_without_mutation(self):
        from tepid_h1.modeling.cache import AttentionCache

        cache = AttentionCache()

        with self.assertRaisesRegex(ValueError, "provided together"):
            cache.load_state_dict(
                {
                    "seq_len": 2,
                    "device": "cpu",
                    "dtype": "float32",
                    "k_cache": torch.randn(1, 2, 3),
                    "v_cache": None,
                }
            )

        self.assertTrue(cache.is_empty)
        self.assertEqual(cache.seq_len, 0)

    def test_load_state_rejects_seq_len_mismatch_without_mutation(self):
        from tepid_h1.modeling.cache import AttentionCache

        cache = AttentionCache()

        with self.assertRaisesRegex(ValueError, "seq_len"):
            cache.load_state_dict(
                {
                    "seq_len": 3,
                    "device": "cpu",
                    "dtype": "float32",
                    "k_cache": torch.randn(1, 2, 3),
                    "v_cache": torch.randn(1, 2, 3),
                }
            )

        self.assertTrue(cache.is_empty)
        self.assertEqual(cache.seq_len, 0)

    def test_load_state_rejects_unknown_dtype_without_mutation(self):
        from tepid_h1.modeling.cache import AttentionCache

        cache = AttentionCache()

        with self.assertRaisesRegex(ValueError, "dtype"):
            cache.load_state_dict(
                {
                    "seq_len": 0,
                    "device": "cpu",
                    "dtype": "float64",
                    "k_cache": None,
                    "v_cache": None,
                }
            )

        self.assertTrue(cache.is_empty)
        self.assertEqual(cache.dtype, torch.float32)

    def test_load_state_rejects_declared_dtype_mismatch_without_mutation(self):
        from tepid_h1.modeling.cache import AttentionCache

        cache = AttentionCache()

        with self.assertRaisesRegex(ValueError, "dtype"):
            cache.load_state_dict(
                {
                    "seq_len": 2,
                    "device": "cpu",
                    "dtype": "float16",
                    "k_cache": torch.randn(1, 2, 3, dtype=torch.float32),
                    "v_cache": torch.randn(1, 2, 3, dtype=torch.float32),
                }
            )

        self.assertTrue(cache.is_empty)
        self.assertEqual(cache.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
