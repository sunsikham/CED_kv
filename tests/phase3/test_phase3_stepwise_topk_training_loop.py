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


class Phase3StepwiseTrainingLoopTest(unittest.TestCase):
    def test_stepwise_topk_and_optimizer_update(self) -> None:
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"torch unavailable: {exc}")
            return

        config = load_config("configs/base.yaml")
        base_settings = resolve_phase3_settings(config=config)
        settings = replace(
            base_settings,
            train_steps=4,
            train_lr=0.2,
            topk_schedule=(4, 2),
            topk_milestones=(0.0, 0.5),
            span_budget=2,
            cvar_tail_fraction=0.5,
        )

        runtime = _FakeRuntime(torch_mod=torch)
        warm_scores = [1.0, 0.8, 0.4, 0.1]
        logits = _init_slot_logits(runtime=runtime, warm_scores=warm_scores, slot_count=settings.span_budget)
        before = logits.detach().clone()
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
        )

        self.assertEqual(len(result["topk_trace"]), settings.train_steps)
        self.assertEqual(len(result["loss_curve"]), settings.train_steps)
        seen_topk = {int(row["topk"]) for row in result["topk_trace"]}
        self.assertEqual(seen_topk, {2, 4})
        self.assertFalse(torch.allclose(before, episode.logits.detach()))


if __name__ == "__main__":
    unittest.main()

