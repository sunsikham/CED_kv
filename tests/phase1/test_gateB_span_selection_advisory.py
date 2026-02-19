from __future__ import annotations

import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase1 import resolve_phase1_settings, run_phase1


class Phase1GateBAdvisoryTest(unittest.TestCase):
    def test_gateb_can_fail_while_gatea_passes(self) -> None:
        config = load_config("configs/base.yaml")
        config["phase1"]["eval"]["on_samples"] = 8
        config["phase1"]["eval"]["off_samples"] = 16
        config["phase1"]["thresholds"]["on_gain_advisory"] = 0.9
        config["phase1"]["select"]["mode"] = "random_span"
        settings = resolve_phase1_settings(config=config)
        result = run_phase1(config=config, seed=7, settings=settings)
        gate_a = result["report"]["gates"]["A"]
        gate_b = result["report"]["gates"]["B"]
        self.assertTrue(gate_a["pass"])
        self.assertFalse(gate_b["pass"])


if __name__ == "__main__":
    unittest.main()

