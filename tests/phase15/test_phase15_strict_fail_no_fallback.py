from __future__ import annotations

import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase1 import resolve_phase1_settings, run_phase1
from cedkv_mvp.model_hf import HFBackendUnavailable


class Phase15StrictFailTest(unittest.TestCase):
    def test_hf_strict_mode_does_not_fallback(self) -> None:
        config = load_config("configs/base.yaml")
        config["phase1"]["runtime"]["backend"] = "hf"
        config["phase1"]["runtime"]["allow_mock_fallback"] = False
        config["phase1"]["model"]["id"] = "this-model/does-not-exist"
        settings = resolve_phase1_settings(config=config)
        with self.assertRaises(HFBackendUnavailable):
            run_phase1(config=config, seed=7, settings=settings)


if __name__ == "__main__":
    unittest.main()

