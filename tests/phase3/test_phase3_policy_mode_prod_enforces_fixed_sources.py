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


class Phase3PolicyModeProdEnforcementTest(unittest.TestCase):
    def test_prod_rejects_nonfixed_constraint_source(self) -> None:
        config = load_config("configs/base.yaml")
        modified = copy.deepcopy(config)
        modified["phase3"]["runtime"]["policy_mode"] = "prod"
        _phase3_off_loss_block(modified)["constraint_source"] = "hard_pool"
        with self.assertRaises(ValueError):
            resolve_phase3_settings(config=modified)

    def test_prod_rejects_nonfixed_dual_metric_source(self) -> None:
        config = load_config("configs/base.yaml")
        modified = copy.deepcopy(config)
        modified["phase3"]["runtime"]["policy_mode"] = "prod"
        modified["phase3"]["train"]["lambda_off"]["dual"]["metric_source"] = "hard_pool"
        with self.assertRaises(ValueError):
            resolve_phase3_settings(config=modified)

    def test_ci_env_forces_prod(self) -> None:
        config = load_config("configs/base.yaml")
        modified = copy.deepcopy(config)
        modified["phase3"]["runtime"]["policy_mode"] = "debug"
        modified["phase3"]["runtime"]["debug_allow_nonfixed_constraint_sources"] = True
        _phase3_off_loss_block(modified)["constraint_source"] = "hard_pool"
        modified["phase3"]["train"]["lambda_off"]["dual"]["metric_source"] = "hard_pool"
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=False):
            with self.assertRaises(ValueError):
                resolve_phase3_settings(config=modified)


if __name__ == "__main__":
    unittest.main()
