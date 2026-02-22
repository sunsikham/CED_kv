from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import should_eval_constraint_tick, update_constraint_warn_state


class Phase3ConstraintWarnEvalTickSemanticsTest(unittest.TestCase):
    def test_eval_tick_follows_eval_every(self) -> None:
        ticks = [step for step in range(7) if should_eval_constraint_tick(step=step, eval_every=3)]
        self.assertEqual(ticks, [0, 3, 6])

    def test_patience_is_counted_in_eval_ticks(self) -> None:
        threshold = 0.05
        margin = 0.01
        patience = 2
        consecutive = 0

        first = update_constraint_warn_state(
            metric_value=0.04,
            threshold=threshold,
            margin=margin,
            patience_eval_ticks=patience,
            consecutive_violations=consecutive,
        )
        self.assertFalse(first["violated"])
        self.assertFalse(first["triggered"])
        consecutive = int(first["consecutive_violations"])

        second = update_constraint_warn_state(
            metric_value=0.07,
            threshold=threshold,
            margin=margin,
            patience_eval_ticks=patience,
            consecutive_violations=consecutive,
        )
        self.assertTrue(second["violated"])
        self.assertFalse(second["triggered"])
        consecutive = int(second["consecutive_violations"])

        third = update_constraint_warn_state(
            metric_value=0.08,
            threshold=threshold,
            margin=margin,
            patience_eval_ticks=patience,
            consecutive_violations=consecutive,
        )
        self.assertTrue(third["violated"])
        self.assertTrue(third["triggered"])
        self.assertEqual(third["consecutive_violations"], 2)


if __name__ == "__main__":
    unittest.main()
