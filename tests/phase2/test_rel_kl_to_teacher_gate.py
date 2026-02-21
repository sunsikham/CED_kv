from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase2 import _gateb_pass


class Phase2GateRuleTest(unittest.TestCase):
    def test_gate_fails_when_teacher_alignment_is_bad(self) -> None:
        passed = _gateb_pass(
            on_gain=0.2,
            on_gain_min=0.02,
            off_delta_p99=0.01,
            off_delta_p99_max=0.05,
            delta_on_mean=0.03,
            delta_on_min=0.01,
            rel_kl_to_teacher=1.4,
            rel_kl_to_teacher_max=1.0,
            on_acc_select=0.8,
            on_acc_random_mean=0.2,
        )
        self.assertFalse(passed)

    def test_gate_passes_with_all_constraints(self) -> None:
        passed = _gateb_pass(
            on_gain=0.1,
            on_gain_min=0.02,
            off_delta_p99=0.02,
            off_delta_p99_max=0.05,
            delta_on_mean=0.02,
            delta_on_min=0.01,
            rel_kl_to_teacher=0.8,
            rel_kl_to_teacher_max=1.0,
            on_acc_select=0.6,
            on_acc_random_mean=0.4,
        )
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()

