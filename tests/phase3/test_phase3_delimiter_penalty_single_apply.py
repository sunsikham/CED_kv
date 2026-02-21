from __future__ import annotations

from dataclasses import replace
import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase3 import (
    _final_slot_weights_from_logits,
    _slot_weights_from_logits_tensor,
    apply_delimiter_mass_constraints,
    resolve_phase3_settings,
)


class _FakeModel:
    def __init__(self, torch_mod: object) -> None:
        self.device = getattr(torch_mod, "device")("cpu")


class _FakeRuntime:
    def __init__(self, torch_mod: object) -> None:
        self.torch = torch_mod
        self.model = _FakeModel(torch_mod=torch_mod)


class Phase3DelimiterPenaltySingleApplyTest(unittest.TestCase):
    def test_final_slot_weights_apply_penalty_once(self) -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"torch unavailable: {exc}")
            return

        config = load_config("configs/base.yaml")
        base_settings = resolve_phase3_settings(config=config)
        settings = replace(
            base_settings,
            delimiter_mass_cap=1.0,
            delimiter_mass_penalty=0.2,
            delimiter_apply_stage="post_topk_renorm_per_slot",
        )
        runtime = _FakeRuntime(torch_mod=torch)
        atom_types = ["delimiter_only", "entry"]
        logits = torch.tensor([[2.0, 0.0]], dtype=torch.float32)

        final_weights = _final_slot_weights_from_logits(
            runtime=runtime,
            logits=logits,
            topk=2,
            atom_types=atom_types,
            settings=settings,
        )[0]
        base_weights = _slot_weights_from_logits_tensor(
            torch_mod=torch,
            logits=logits,
            topk=2,
            atom_types=atom_types,
        ).detach().cpu().tolist()[0]
        expected = apply_delimiter_mass_constraints(
            weights=[float(value) for value in base_weights],
            atom_types=atom_types,
            mass_cap=settings.delimiter_mass_cap,
            mass_penalty=settings.delimiter_mass_penalty,
        )

        self.assertAlmostEqual(final_weights[0], expected[0], places=7)
        self.assertAlmostEqual(final_weights[1], expected[1], places=7)
        self.assertAlmostEqual(sum(final_weights), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
