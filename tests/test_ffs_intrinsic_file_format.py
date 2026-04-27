from __future__ import annotations

import unittest

import numpy as np

from data_process.depth_backends.geometry import format_ffs_intrinsic_text


class FfsIntrinsicFileFormatTest(unittest.TestCase):
    def test_writes_expected_two_line_format(self) -> None:
        K = np.array(
            [
                [430.59, 0.0, 432.38904],
                [0.0, 430.59, 239.78023],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        text = format_ffs_intrinsic_text(K, 0.0950430259)
        lines = text.strip().splitlines()

        self.assertEqual(len(lines), 2)
        first_line_values = np.fromstring(lines[0], sep=" ")
        np.testing.assert_allclose(first_line_values, K.reshape(-1), rtol=0, atol=1e-7)
        self.assertEqual(lines[1], "0.09504303")


if __name__ == "__main__":
    unittest.main()
