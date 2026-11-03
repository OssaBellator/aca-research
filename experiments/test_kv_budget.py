"""Small deterministic correctness checks, independent of GPU availability."""
import unittest

from kv_budget import kv_bytes, fmt_gib


class KVMathTests(unittest.TestCase):
    def test_48_kib_per_token(self):
        self.assertEqual(kv_bytes(24, 8, 64, 1, 2), 48 * 1024)

    def test_6_gib_at_128k(self):
        self.assertEqual(kv_bytes(24, 8, 64, 131072, 2), 6 * 1024**3)

    def test_128_kib_per_token(self):
        self.assertEqual(kv_bytes(32, 8, 128, 1, 2), 128 * 1024)

    def test_4_gib_at_32k(self):
        self.assertEqual(kv_bytes(32, 8, 128, 32768, 2), 4 * 1024**3)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            kv_bytes(0, 8, 64, 4096, 2)
        with self.assertRaises(ValueError):
            kv_bytes(24, -1, 64, 4096, 2)

    def test_gib_format(self):
        self.assertEqual(fmt_gib(1024**3), "1.000 GiB")


if __name__ == "__main__":
    unittest.main()
