from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import select_constraint_p99


class Phase3ConstraintMetricSourceTest(unittest.TestCase):
    def test_fixed_eval_metric_source_ignores_hard_pool(self) -> None:
        selected = select_constraint_p99(
            metric_source="fixed_stress_eval",
            fixed_p99=0.2,
            hard_pool_p99=99.0,
        )
        self.assertEqual(selected, 0.2)

    def test_hard_pool_metric_source_uses_hard_pool(self) -> None:
        selected = select_constraint_p99(
            metric_source="hard_pool",
            fixed_p99=0.2,
            hard_pool_p99=3.5,
        )
        self.assertEqual(selected, 3.5)


if __name__ == "__main__":
    unittest.main()

