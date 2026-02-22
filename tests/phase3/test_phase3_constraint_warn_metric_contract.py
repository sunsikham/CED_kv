from __future__ import annotations

from dataclasses import replace
import unittest

from cedkv_mvp.config import load_config
from cedkv_mvp.eval_phase3 import (
    _EpisodeTrainState,
    _init_slot_logits,
    _optimize_episode_states,
    resolve_phase3_settings,
)


class _FakeModel:
    def __init__(self, torch_mod: object) -> None:
        self.device = getattr(torch_mod, "device")("cpu")


class _FakeRuntime:
    def __init__(self, torch_mod: object) -> None:
        self.torch = torch_mod
        self.model = _FakeModel(torch_mod=torch_mod)


class Phase3ConstraintWarnMetricContractTest(unittest.TestCase):
    def test_warn_metric_uses_fixed_stress_source(self) -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"torch unavailable: {exc}")
            return

        config = load_config("configs/base.yaml")
        base_settings = resolve_phase3_settings(config=config)
        settings = replace(
            base_settings,
            train_steps=3,
            span_budget=2,
            topk_schedule=(4,),
            topk_milestones=(0.0,),
            cvar_tail_fraction=1.0,
            off_hard_pool_size=1,
            off_constraint_source="hard_pool",
            dual_metric_source="hard_pool",
            constraint_warn_enabled=True,
            constraint_eval_every=1,
            constraint_warn_patience_eval_ticks=2,
            constraint_warn_margin=0.0,
            lambda_off_adaptive=False,
        )
        runtime = _FakeRuntime(torch_mod=torch)
        warm_scores = [1.0, 0.8, 0.4, 0.1]
        logits = _init_slot_logits(runtime=runtime, warm_scores=warm_scores, slot_count=settings.span_budget)
        episode = _EpisodeTrainState(
            sample={"sample_id": "s0", "episode_id": "e0"},
            atoms=[
                {"atom_type": "entry", "text": "key_0 -> V0\n"},
                {"atom_type": "key_only", "text": "key_0\n"},
                {"atom_type": "value_only", "text": "V0\n"},
                {"atom_type": "delimiter_only", "text": " -> "},
            ],
            atom_types=["entry", "key_only", "value_only", "delimiter_only"],
            warm_scores=warm_scores,
            warm_anchor_ids=[0, 1],
            logits=logits,
        )

        result = _optimize_episode_states(
            runtime=runtime,
            episode_states=[episode],
            settings=settings,
            off_samples_train=[{"sample_id": "train_a"}],
            off_samples_fixed=[{"sample_id": "fixed_a"}],
        )

        self.assertGreaterEqual(len(result["constraint_warn_trace"]), 1)
        for row in result["constraint_warn_trace"]:
            self.assertEqual(row["metric_name"], "off_delta_p99_stress_proxy")
            self.assertEqual(row["metric_source"], "fixed_stress_eval")


if __name__ == "__main__":
    unittest.main()
