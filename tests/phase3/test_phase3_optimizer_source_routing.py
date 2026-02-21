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


def _make_episode(runtime: object, settings: object) -> object:
    warm_scores = [1.0, 0.8, 0.4, 0.1]
    logits = _init_slot_logits(runtime=runtime, warm_scores=warm_scores, slot_count=settings.span_budget)
    return _EpisodeTrainState(
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


class Phase3OptimizerSourceRoutingTest(unittest.TestCase):
    def test_dual_and_constraint_sources_flow_into_optimizer_metrics(self) -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"torch unavailable: {exc}")
            return

        config = load_config("configs/base.yaml")
        base_settings = resolve_phase3_settings(config=config)
        settings = replace(
            base_settings,
            train_steps=2,
            train_lr=0.1,
            topk_schedule=(4,),
            topk_milestones=(0.0,),
            span_budget=2,
            cvar_tail_fraction=1.0,
            off_hard_pool_size=1,
            off_train_source="hard_pool",
            off_constraint_source="fixed_stress_eval",
            dual_metric_source="hard_pool",
            lambda_off_adaptive=False,
        )

        runtime = _FakeRuntime(torch_mod=torch)
        episode = _make_episode(runtime=runtime, settings=settings)
        result = _optimize_episode_states(
            runtime=runtime,
            episode_states=[episode],
            settings=settings,
            off_samples_train=[
                {"sample_id": "train_a", "query_text": "alpha"},
                {"sample_id": "train_b", "query_text": "beta"},
                {"sample_id": "train_c", "query_text": "gamma"},
            ],
            off_samples_fixed=[
                {"sample_id": "fixed_a", "query_text": "fixed"},
            ],
        )

        row = result["loss_curve"][0]
        self.assertEqual(row["off_train_source"], "hard_pool")
        self.assertEqual(row["off_constraint_source"], "fixed_stress_eval")
        self.assertEqual(row["dual_metric_source"], "hard_pool")
        self.assertAlmostEqual(row["off_p99_constraint"], row["off_p99_fixed"])
        self.assertAlmostEqual(row["proxy_off_p99"], row["off_p99_hard_pool"])

    def test_off_train_source_switch_changes_loss_off_path(self) -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"torch unavailable: {exc}")
            return

        config = load_config("configs/base.yaml")
        base_settings = resolve_phase3_settings(config=config)
        shared = dict(
            train_steps=1,
            train_lr=0.1,
            topk_schedule=(4,),
            topk_milestones=(0.0,),
            span_budget=2,
            cvar_tail_fraction=1.0,
            off_hard_pool_size=1,
            off_constraint_source="fixed_stress_eval",
            dual_metric_source="fixed_stress_eval",
            lambda_off_adaptive=False,
        )
        hard_settings = replace(base_settings, off_train_source="hard_pool", **shared)
        fixed_settings = replace(base_settings, off_train_source="fixed_stress_eval", **shared)

        runtime = _FakeRuntime(torch_mod=torch)
        hard_episode = _make_episode(runtime=runtime, settings=hard_settings)
        fixed_episode = _make_episode(runtime=runtime, settings=fixed_settings)
        off_train = [
            {"sample_id": "train_a", "query_text": "alpha"},
            {"sample_id": "train_b", "query_text": "beta"},
            {"sample_id": "train_c", "query_text": "gamma"},
        ]
        off_fixed = [{"sample_id": "fixed_a", "query_text": "fixed"}]

        hard_result = _optimize_episode_states(
            runtime=runtime,
            episode_states=[hard_episode],
            settings=hard_settings,
            off_samples_train=off_train,
            off_samples_fixed=off_fixed,
        )
        fixed_result = _optimize_episode_states(
            runtime=runtime,
            episode_states=[fixed_episode],
            settings=fixed_settings,
            off_samples_train=off_train,
            off_samples_fixed=off_fixed,
        )

        hard_loss_off = hard_result["loss_curve"][0]["loss_off"]
        fixed_loss_off = fixed_result["loss_curve"][0]["loss_off"]
        self.assertNotAlmostEqual(hard_loss_off, fixed_loss_off, places=6)


if __name__ == "__main__":
    unittest.main()
