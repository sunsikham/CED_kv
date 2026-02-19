from __future__ import annotations

import os
import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase1 import resolve_phase1_settings, run_phase1
from cedkv_mvp.model_hf import HFBackendUnavailable


RUN_HF_TESTS = os.environ.get("RUN_HF_TESTS") == "1"


@unittest.skipUnless(RUN_HF_TESTS, "Set RUN_HF_TESTS=1 to enable HF Phase1.5 tests.")
class Phase15HfSmokeTest(unittest.TestCase):
    def test_hf_gatea_smoke(self) -> None:
        config = load_config("configs/base.yaml")
        config["phase1"]["runtime"]["backend"] = "hf"
        config["phase1"]["runtime"]["allow_mock_fallback"] = False
        config["phase1"]["eval"]["on_samples"] = 2
        config["phase1"]["eval"]["off_samples"] = 2
        settings = resolve_phase1_settings(config=config)
        try:
            result = run_phase1(config=config, seed=7, settings=settings)
        except HFBackendUnavailable as exc:
            self.skipTest(f"HF runtime unavailable for smoke: {exc}")
        self.assertIn("A", result["report"]["gates"])
        self.assertEqual(result["report"]["gateA_source"], "hf_logits")


if __name__ == "__main__":
    unittest.main()

