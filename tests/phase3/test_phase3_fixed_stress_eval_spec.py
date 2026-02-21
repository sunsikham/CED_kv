from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase3 import FixedStressEvalSpec, build_fixed_stress_eval_samples


class Phase3FixedStressEvalSpecTest(unittest.TestCase):
    def test_fixed_stress_eval_is_seed_deterministic(self) -> None:
        spec = FixedStressEvalSpec(
            size=8,
            seed=3407,
            sampling="once_per_experiment",
            definition="phase2_stress_compatible",
        )
        first = build_fixed_stress_eval_samples(spec)
        second = build_fixed_stress_eval_samples(spec)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)

    def test_fixed_stress_eval_uses_stable_ids(self) -> None:
        spec = FixedStressEvalSpec(
            size=3,
            seed=99,
            sampling="once_per_experiment",
            definition="phase2_stress_compatible",
        )
        samples = build_fixed_stress_eval_samples(spec)
        self.assertEqual(samples[0]["sample_id"], "fixed_off_99_0")
        self.assertEqual(samples[1]["sample_id"], "fixed_off_99_1")
        self.assertEqual(samples[2]["sample_id"], "fixed_off_99_2")


if __name__ == "__main__":
    unittest.main()

