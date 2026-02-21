from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import evaluate_phase3_outcome


class Phase3SuccessRuleTest(unittest.TestCase):
    def test_gate_and_milestone_are_separate(self) -> None:
        result = evaluate_phase3_outcome(
            gatea_pass=False,
            gateb_pass=False,
            explore_enabled=True,
            run_index=1,
            max_runs=5,
            on_gain_drop=0.0,
            off_tail_improve_ratio=0.5,
            on_gain_drop_max=0.01,
            off_tail_improve_min=0.2,
        )
        self.assertFalse(result["gate_pass"])
        self.assertTrue(result["milestone_pass"])

    def test_milestone_requires_explore_window(self) -> None:
        result = evaluate_phase3_outcome(
            gatea_pass=True,
            gateb_pass=True,
            explore_enabled=True,
            run_index=7,
            max_runs=5,
            on_gain_drop=0.0,
            off_tail_improve_ratio=0.5,
            on_gain_drop_max=0.01,
            off_tail_improve_min=0.2,
        )
        self.assertFalse(result["milestone_applicable"])
        self.assertFalse(result["milestone_pass"])
        self.assertTrue(result["gate_pass"])


if __name__ == "__main__":
    unittest.main()

