"""Tests for the KV-cache implementation."""
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AttentionCacheTests(unittest.TestCase):
    def setUp(self):
        from tepid_h1.modeling.cache import AttentionCache

        self.Cache = AttentionCache

    def test_empty_cache_default_state(self):
        cache = self.Cache()
        self.assertTrue(cache.is_empty)
        self.assertEqual(cache.seq_len, 0)
        self.assertEqual(cache.device.type, "cpu")

    def test_empty_cache_to_returns_same(self):
        cache = self.Cache()
        same = cache.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.assertIs(same, cache)

    def test_mismatched_kv_raises(self):
        with self.assertRaisesRegex(ValueError, "k_cache requires v_cache"):
            self.Cache(
                k_cache=torch.empty(1, 2, 4),
                v_cache=None,
            )
        with self.assertRaisesRegex(ValueError, "v_cache requires k_cache"):
            self.Cache(
                k_cache=None,
                v_cache=torch.empty(1, 2, 4),
            )

    def test_initial_update_uses_provided_tensors(self):
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        cache = self.Cache()
        returned_k, returned_v = cache.update(k, v)

        self.assertIs(returned_k, k)
        self.assertIs(returned_v, v)
        self.assertEqual(cache.seq_len, 4)
        self.assertFalse(cache.is_empty)

    def test_concatenation_expands_sequence(self):
        k1 = torch.randn(1, 3, 4, 8)
        v1 = torch.randn(1, 3, 4, 8)
        k2 = torch.randn(1, 2, 4, 8)
        v2 = torch.randn(1, 2, 4, 8)

        cache = self.Cache()
        cache.update(k1, v1)
        new_k, new_v = cache.update(k2, v2)

        self.assertEqual(new_k.shape, (1, 5, 4, 8))
        self.assertEqual(new_v.shape, (1, 5, 4, 8))
        self.assertEqual(cache.seq_len, 5)

    def test_truncation_respects_cache_length(self):
        k = torch.randn(1, 10, 4, 8)
        v = torch.randn(1, 10, 4, 8)
        cache = self.Cache()
        cache.update(k, v)

        new_k = torch.randn(1, 5, 4, 8)
        new_v = torch.randn(1, 5, 4, 8)
        returned_k, returned_v = cache.update(new_k, new_v, cache_length=10)

        self.assertEqual(returned_k.shape[1], 10)
        self.assertEqual(returned_k.shape[1], returned_v.shape[1])

    def test_truncation_returns_original_when_full(self):
        k = torch.randn(1, 10, 4, 8)
        v = torch.randn(1, 10, 4, 8)
        cache = self.Cache()
        cache.update(k, v)

        new_k = torch.randn(1, 5, 4, 8)
        new_v = torch.randn(1, 5, 4, 8)
        returned_k, returned_v = cache.update(new_k, new_v, cache_length=5)

        self.assertIs(returned_k, k)
        self.assertIs(returned_v, v)

    def test_incompatible_shapes_raise(self):
        cache = self.Cache()
        with self.assertRaisesRegex(ValueError, "key and value sequences"):
            cache.update(torch.empty(1, 4, 8, 16), torch.empty(1, 3, 8, 16))
        with self.assertRaisesRegex(ValueError, "expected at least 3D tensors"):
            cache.update(torch.empty(1, 4), torch.empty(1, 4))
        with self.assertRaisesRegex(ValueError, "batch dimensions must match"):
            cache.update(torch.empty(1, 4, 8, 16), torch.empty(2, 4, 8, 16))
        with self.assertRaisesRegex(ValueError, "dtypes must match"):
            cache.update(torch.empty(1, 4, 8, 16, dtype=torch.float32), torch.empty(1, 4, 8, 16, dtype=torch.float16))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            cache.update(torch.empty(1, 4, 8, 16), torch.empty(1, 4, 8, 16), cache_length=0)
        with self.assertRaisesRegex(ValueError, "must accommodate"):
            cache.update(torch.empty(1, 10, 8, 16), torch.empty(1, 10, 8, 16), cache_length=5)

    def test_reset_clears_cache(self):
        cache = self.Cache()
        cache.update(torch.empty(1, 4, 8, 16), torch.empty(1, 4, 8, 16))
        cache.reset()
        self.assertTrue(cache.is_empty)
        self.assertEqual(cache.seq_len, 0)

    def test_state_dict_round_trip(self):
        cache = self.Cache()
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        cache = self.Cache()
        cache.update(k, v)

        state = cache.state_dict
        restored = self.Cache()
        restored.load_state_dict(state)

        self.assertEqual(restored.seq_len, 4)
        self.assertEqual(restored.device.type, "cpu")
        torch.testing.assert_close(restored.k_cache, k)
        torch.testing.assert_close(restored.v_cache, v)

    def test_load_state_dict_validates_sequence_length(self):
        with self.assertRaisesRegex(ValueError, "seq_len"):
            self.Cache().load_state_dict({"seq_len": -1})
        with self.assertRaisesRegex(ValueError, "seq_len"):
            self.Cache().load_state_dict({"seq_len": True})

    def test_load_state_dict_validates_kv_dimensions(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            self.Cache().load_state_dict({
                "seq_len": 4,
                "device": "cpu",
                "dtype": "float32",
                "k_cache": torch.empty(1, 4, 8, 16),
                "v_cache": torch.empty(1, 3, 8, 16),
            })

    def test_cache_dtype_float16(self):
        cache = self.Cache()
        cache.load_state_dict({
            "seq_len": 0,
            "device": "cpu",
            "dtype": "float16",
            "k_cache": None,
            "v_cache": None,
        })
        self.assertEqual(cache.dtype, torch.float16)

    def test_cache_dtype_bfloat16(self):
        cache = self.Cache()
        cache.load_state_dict({
            "seq_len": 0,
            "device": "cpu",
            "dtype": "bfloat16",
            "k_cache": None,
            "v_cache": None,
        })
        self.assertEqual(cache.dtype, torch.bfloat16)

    def test_device_dtype_derived_from_provided_tensors(self):
        k = torch.randn(1, 4, 8, 16, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 16, dtype=torch.float16)
        cache = self.Cache()
        cache.update(k, v)
        self.assertEqual(cache.dtype, torch.float16)
        self.assertEqual(cache.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
