from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase2 import _candidate_mask_for_policy, _select_spans_with_policy


class Phase2CandidatePolicyTest(unittest.TestCase):
    def test_value_tokens_are_not_candidates(self) -> None:
        text = "Demo:\nkey_0 -> V0\nkey_1 -> V1\n\n"
        token_spans = [
            (6, 11),   # key_0
            (12, 14),  # ->
            (15, 17),  # V0
            (18, 23),  # key_1
            (24, 26),  # ->
            (27, 29),  # V1
        ]
        mask = _candidate_mask_for_policy(
            prefill_text=text,
            token_spans=token_spans,
            policy="input_and_delimiter_only",
        )
        self.assertEqual(mask, [True, True, False, True, True, False])

    def test_center_policy_blocks_span_start(self) -> None:
        scores = [0.9, 0.8, 0.1, 0.1]
        candidate_mask = [False, True, True, True]
        spans = _select_spans_with_policy(
            scores=scores,
            candidate_mask=candidate_mask,
            span_len=2,
            span_budget=1,
            min_distance=0,
            center_policy="start",
        )
        self.assertEqual(spans, [(1, 3)])


if __name__ == "__main__":
    unittest.main()

