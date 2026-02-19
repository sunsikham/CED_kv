from __future__ import annotations

import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase1 import resolve_phase1_settings, run_phase1


class Phase1GateARoundtripTest(unittest.TestCase):
    def test_roundtrip_relative_kl_condition_holds(self) -> None:
        config = load_config("configs/base.yaml")
        config["phase1"]["eval"]["on_samples"] = 8
        config["phase1"]["eval"]["off_samples"] = 16
        settings = resolve_phase1_settings(config=config)
        result = run_phase1(config=config, seed=7, settings=settings)
        gate_a = result["report"]["gates"]["A"]
        self.assertTrue(gate_a["relative_roundtrip_ok"])
        self.assertLessEqual(
            gate_a["roundtrip_student_teacher_kl"],
            gate_a["alpha_roundtrip"] * gate_a["roundtrip_base_teacher_kl"],
        )


if __name__ == "__main__":
    unittest.main()

