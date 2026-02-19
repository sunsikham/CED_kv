from __future__ import annotations

import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase1 import resolve_phase1_settings, run_phase1


class Phase1GateATeacherSanityTest(unittest.TestCase):
    def test_low_teacher_divergence_fails_gatea(self) -> None:
        config = load_config("configs/base.yaml")
        config["phase1"]["eval"]["on_samples"] = 8
        config["phase1"]["eval"]["off_samples"] = 16
        config["phase1"]["mock"]["teacher_answer_prob"] = 0.08
        config["phase1"]["mock"]["teacher_default_prob"] = 0.88
        config["phase1"]["mock"]["teacher_steer_prob"] = 0.02
        config["phase1"]["thresholds"]["teacher_min_divergence"] = 0.1
        settings = resolve_phase1_settings(config=config)
        result = run_phase1(config=config, seed=7, settings=settings)
        gate_a = result["report"]["gates"]["A"]
        self.assertLess(gate_a["teacher_divergence_mean"], gate_a["teacher_min_divergence"])
        self.assertFalse(gate_a["pass"])


if __name__ == "__main__":
    unittest.main()
