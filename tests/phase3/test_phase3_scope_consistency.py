from __future__ import annotations

from dataclasses import replace
import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase3 import _validate_scope_consistency, resolve_phase3_settings


class Phase3ScopeConsistencyTest(unittest.TestCase):
    def test_scope_consistency_accepts_valid_default(self) -> None:
        settings = resolve_phase3_settings(config=load_config("configs/base.yaml"))
        result = _validate_scope_consistency(settings=settings)
        self.assertTrue(result["checked"])
        self.assertTrue(result["consistent"])

    def test_scope_consistency_rejects_invalid_scope(self) -> None:
        settings = resolve_phase3_settings(config=load_config("configs/base.yaml"))
        broken = replace(settings, layer_scope_injection="invalid_scope")
        with self.assertRaises(ValueError):
            _validate_scope_consistency(settings=broken)


if __name__ == "__main__":
    unittest.main()

