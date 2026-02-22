from __future__ import annotations

import copy
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


class Phase3TrainSourceHardPoolAllowedInProdTest(unittest.TestCase):
    def test_train_source_hard_pool_is_allowed_in_prod(self) -> None:
        config = load_config("configs/base.yaml")
        modified = copy.deepcopy(config)
        modified["phase3"]["runtime"]["policy_mode"] = "prod"
        off_block = _phase3_off_loss_block(modified)
        off_block["train_source"] = "hard_pool"
        off_block["constraint_source"] = "fixed_stress_eval"
        modified["phase3"]["train"]["lambda_off"]["dual"]["metric_source"] = "fixed_stress_eval"
        settings = resolve_phase3_settings(config=modified)
        self.assertEqual(settings.policy_mode_effective, "prod")
        self.assertEqual(settings.off_train_source, "hard_pool")
        self.assertEqual(settings.off_constraint_source, "fixed_stress_eval")
        self.assertEqual(settings.dual_metric_source, "fixed_stress_eval")


if __name__ == "__main__":
    unittest.main()
