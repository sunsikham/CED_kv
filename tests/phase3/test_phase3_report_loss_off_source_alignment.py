from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import select_report_off_loss


class Phase3ReportLossOffSourceAlignmentTest(unittest.TestCase):
    def test_hard_pool_source_uses_hard_pool_loss(self) -> None:
        selected = select_report_off_loss(
            train_source="hard_pool",
            hard_pool_loss=1.23,
            fixed_stress_loss=9.99,
        )
        self.assertEqual(selected, 1.23)

    def test_fixed_stress_source_uses_fixed_stress_loss(self) -> None:
        selected = select_report_off_loss(
            train_source="fixed_stress_eval",
            hard_pool_loss=1.23,
            fixed_stress_loss=0.42,
        )
        self.assertEqual(selected, 0.42)


if __name__ == "__main__":
    unittest.main()
