from __future__ import annotations

import copy
import os
from unittest import mock
import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase3 import resolve_phase3_settings


def _phase3_off_loss_block(config: dict) -> dict:
    loss = config["phase3"]["loss"]
    if "off" in loss and isinstance(loss["off"], dict):
        return loss["off"]
    if False in loss and isinstance(loss[False], dict):
        return loss[False]
    loss["off"] = {}
    return loss["off"]


class Phase3PolicyModeDebugAllowNonfixedTest(unittest.TestCase):
    def test_debug_requires_explicit_override_for_nonfixed(self) -> None:
        config = load_config("configs/base.yaml")
        modified = copy.deepcopy(config)
        modified["phase3"]["runtime"]["policy_mode"] = "debug"
        modified["phase3"]["runtime"]["debug_allow_nonfixed_constraint_sources"] = False
        _phase3_off_loss_block(modified)["constraint_source"] = "hard_pool"
        with mock.patch.dict(os.environ, {"CI": ""}, clear=False):
            with self.assertRaises(ValueError):
                resolve_phase3_settings(config=modified)

    def test_debug_allows_nonfixed_when_override_enabled(self) -> None:
        config = load_config("configs/base.yaml")
        modified = copy.deepcopy(config)
        modified["phase3"]["runtime"]["policy_mode"] = "debug"
        modified["phase3"]["runtime"]["debug_allow_nonfixed_constraint_sources"] = True
        _phase3_off_loss_block(modified)["constraint_source"] = "hard_pool"
        modified["phase3"]["train"]["lambda_off"]["dual"]["metric_source"] = "hard_pool"
        with mock.patch.dict(os.environ, {"CI": ""}, clear=False):
            settings = resolve_phase3_settings(config=modified)
        self.assertEqual(settings.policy_mode, "debug")
        self.assertEqual(settings.policy_mode_effective, "debug")
        self.assertTrue(settings.debug_allow_nonfixed_constraint_sources)
        self.assertEqual(settings.off_constraint_source, "hard_pool")
        self.assertEqual(settings.dual_metric_source, "hard_pool")


if __name__ == "__main__":
    unittest.main()
