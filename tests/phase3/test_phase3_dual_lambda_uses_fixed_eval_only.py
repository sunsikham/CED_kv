from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import select_constraint_p99


class Phase3DualLambdaFixedEvalOnlyTest(unittest.TestCase):
    def test_dual_metric_uses_fixed_eval_when_configured(self) -> None:
        selected = select_constraint_p99(
            metric_source="fixed_stress_eval",
            fixed_p99=0.12,
            hard_pool_p99=9.99,
        )
        self.assertEqual(selected, 0.12)


if __name__ == "__main__":
    unittest.main()

