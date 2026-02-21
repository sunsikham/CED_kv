from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import cvar_tail_mean


class Phase3CvarTest(unittest.TestCase):
    def test_tail_fraction_top_20_percent_mean(self) -> None:
        values = [1.0, 2.0, 3.0, 10.0, 20.0]
        # tail_fraction=0.2 -> ceil(5*0.2)=1 -> max value only.
        self.assertAlmostEqual(cvar_tail_mean(values, 0.2), 20.0)

    def test_tail_fraction_top_40_percent_mean(self) -> None:
        values = [1.0, 2.0, 3.0, 10.0, 20.0]
        # tail_fraction=0.4 -> ceil(5*0.4)=2 -> mean(10,20)=15
        self.assertAlmostEqual(cvar_tail_mean(values, 0.4), 15.0)


if __name__ == "__main__":
    unittest.main()

