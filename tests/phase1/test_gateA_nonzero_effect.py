from __future__ import annotations

import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase1 import resolve_phase1_settings, run_phase1


class Phase1GateANonzeroEffectTest(unittest.TestCase):
    def test_gatea_nonnull_effect_exceeds_threshold(self) -> None:
        config = load_config("configs/base.yaml")
        config["phase1"]["eval"]["on_samples"] = 8
        config["phase1"]["eval"]["off_samples"] = 16
        settings = resolve_phase1_settings(config=config)
        result = run_phase1(config=config, seed=7, settings=settings)
        gate_a = result["report"]["gates"]["A"]
        self.assertGreaterEqual(gate_a["delta_on_nonnull_mean"], gate_a["eps_nonzero"])


if __name__ == "__main__":
    unittest.main()

