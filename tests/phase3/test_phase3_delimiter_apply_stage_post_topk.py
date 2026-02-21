from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import apply_delimiter_mass_constraints


class Phase3DelimiterMassConstraintTest(unittest.TestCase):
    def test_delimiter_mass_is_capped(self) -> None:
        weights = [0.6, 0.4]
        atom_types = ["delimiter_only", "entry"]
        constrained = apply_delimiter_mass_constraints(
            weights=weights,
            atom_types=atom_types,
            mass_cap=0.15,
            mass_penalty=0.0,
        )
        self.assertLessEqual(constrained[0], 0.1500001)
        self.assertAlmostEqual(sum(constrained), 1.0)

    def test_penalty_reduces_delimiter_before_cap(self) -> None:
        weights = [0.1, 0.9]
        atom_types = ["delimiter_only", "entry"]
        constrained = apply_delimiter_mass_constraints(
            weights=weights,
            atom_types=atom_types,
            mass_cap=0.5,
            mass_penalty=0.5,
        )
        self.assertLess(constrained[0], 0.1)


if __name__ == "__main__":
    unittest.main()

