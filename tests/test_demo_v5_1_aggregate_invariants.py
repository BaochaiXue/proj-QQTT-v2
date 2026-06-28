from __future__ import annotations

import unittest

import numpy as np


class DemoV51AggregateInvariantsTest(unittest.TestCase):
    def test_generic_array_match_helper_is_not_exposed(self) -> None:
        from demo_v5_1 import chunked_final_data_aggregate as aggregate

        self.assertFalse(hasattr(aggregate, "_arrays_match"))
        self.assertFalse(hasattr(aggregate, "_require_matching_value"))

    def test_static_invariant_normalizer_name_describes_compare_purpose(self) -> None:
        from demo_v5_1 import chunked_final_data_aggregate as aggregate

        self.assertFalse(hasattr(aggregate, "_static_invariant_value"))
        self.assertTrue(
            hasattr(aggregate, "_normalize_static_invariant_for_compare")
        )

    def test_scalar_string_and_singleton_array_do_not_match(self) -> None:
        from demo_v5_1 import chunked_final_data_aggregate as aggregate

        with self.assertRaisesRegex(ValueError, "aggregate invariant mismatch"):
            aggregate._require_matching_scalar_invariant(
                "metadata.json runtime_contract",
                "contract_v1",
                np.asarray(["contract_v1"]),
            )

    def test_array_dtype_mismatch_does_not_match(self) -> None:
        from demo_v5_1 import chunked_final_data_aggregate as aggregate

        with self.assertRaisesRegex(ValueError, "aggregate invariant mismatch"):
            aggregate._require_matching_array_invariant(
                "final_data.pkl query_ids",
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([1, 2], dtype=np.int32),
            )


if __name__ == "__main__":
    unittest.main()
