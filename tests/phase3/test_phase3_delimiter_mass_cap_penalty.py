from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import apply_delimiter_mass_constraints


class Phase3DelimiterMassCapPenaltyTest(unittest.TestCase):
    def test_delimiter_mass_cap_and_penalty(self) -> None:
        constrained = apply_delimiter_mass_constraints(
            weights=[0.7, 0.3],
            atom_types=["delimiter_only", "entry"],
            mass_cap=0.15,
            mass_penalty=0.2,
        )
        self.assertLessEqual(constrained[0], 0.1500001)
        self.assertAlmostEqual(sum(constrained), 1.0)


if __name__ == "__main__":
    unittest.main()

