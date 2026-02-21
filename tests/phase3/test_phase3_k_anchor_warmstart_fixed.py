from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import select_k_anchor_index


class Phase3KAnchorPolicyTest(unittest.TestCase):
    def test_warm_start_fixed_uses_warm_start_index(self) -> None:
        weights = [0.05, 0.8, 0.15]
        self.assertEqual(
            select_k_anchor_index(weights=weights, warm_start_index=0, policy="warm_start_fixed"),
            0,
        )

    def test_warm_start_fixed_fallbacks_when_index_invalid(self) -> None:
        weights = [0.05, 0.8, 0.15]
        self.assertEqual(
            select_k_anchor_index(weights=weights, warm_start_index=99, policy="warm_start_fixed"),
            1,
        )


if __name__ == "__main__":
    unittest.main()

