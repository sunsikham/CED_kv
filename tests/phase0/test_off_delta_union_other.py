from __future__ import annotations

import unittest

from cedkv_mvp.metrics_phase0 import kl_student_to_base_union_other


class OffDeltaUnionOtherTest(unittest.TestCase):
    def test_kl_zero_for_identical_sparse_distributions(self) -> None:
        student = {"A": 0.6, "B": 0.3}
        base = {"A": 0.6, "B": 0.3}
        value = kl_student_to_base_union_other(student_sparse=student, base_sparse=base)
        self.assertAlmostEqual(value, 0.0, places=8)

    def test_kl_positive_for_different_sparse_distributions(self) -> None:
        student = {"A": 0.9}
        base = {"B": 0.9}
        value = kl_student_to_base_union_other(student_sparse=student, base_sparse=base)
        self.assertGreater(value, 0.0)


if __name__ == "__main__":
    unittest.main()

