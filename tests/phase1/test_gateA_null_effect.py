from __future__ import annotations

import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase1 import resolve_phase1_settings, run_phase1


class Phase1GateANullEffectTest(unittest.TestCase):
    def test_gatea_null_effect_is_bounded_by_eps(self) -> None:
        config = load_config("configs/base.yaml")
        config["phase1"]["eval"]["on_samples"] = 8
        config["phase1"]["eval"]["off_samples"] = 16
        settings = resolve_phase1_settings(config=config)
        result = run_phase1(config=config, seed=7, settings=settings)
        gate_a = result["report"]["gates"]["A"]
        self.assertLessEqual(gate_a["delta_on_null_mean"], gate_a["eps_null"])
        self.assertLessEqual(gate_a["delta_off_null_mean"], gate_a["eps_null"])


if __name__ == "__main__":
    unittest.main()

