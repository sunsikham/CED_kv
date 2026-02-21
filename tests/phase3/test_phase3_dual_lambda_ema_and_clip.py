from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import run_dual_lambda_update_trace


class Phase3DualLambdaTest(unittest.TestCase):
    def test_dual_lambda_respects_clip_and_bounds(self) -> None:
        trace = run_dual_lambda_update_trace(
            steps=6,
            lambda_init=1.0,
            eta=10.0,
            threshold=0.05,
            p99_raw_values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ema_beta=0.9,
            update_every=1,
            lambda_min=0.0,
            lambda_max=1.2,
            delta_clip=0.05,
            adaptive=True,
        )
        self.assertTrue(all(0.0 <= row["lambda_off"] <= 1.2 for row in trace))
        self.assertTrue(all(abs(row["delta"]) <= 0.05 for row in trace))
        self.assertAlmostEqual(trace[-1]["lambda_off"], 1.2)

    def test_dual_lambda_update_every_controls_update_period(self) -> None:
        trace = run_dual_lambda_update_trace(
            steps=5,
            lambda_init=1.0,
            eta=1.0,
            threshold=0.05,
            p99_raw_values=[1.0, 1.0, 1.0, 1.0, 1.0],
            ema_beta=0.0,
            update_every=2,
            lambda_min=0.0,
            lambda_max=10.0,
            delta_clip=10.0,
            adaptive=True,
        )
        deltas = [row["delta"] for row in trace]
        self.assertNotEqual(deltas[0], 0.0)
        self.assertEqual(deltas[1], 0.0)
        self.assertNotEqual(deltas[2], 0.0)


if __name__ == "__main__":
    unittest.main()

