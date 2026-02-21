from __future__ import annotations

import unittest

from cedkv_mvp.eval_phase2 import _token_ids_from_spans


class Phase2PrefixTokenTest(unittest.TestCase):
    def test_selected_tokens_respect_prefix_len(self) -> None:
        token_ids = [10, 11, 12, 13, 14]
        spans = [(1, 4)]  # 11,12,13
        selected = _token_ids_from_spans(token_ids=token_ids, spans=spans, prefix_len=2)
        self.assertEqual(selected, [11, 12])

    def test_zero_prefix_len_returns_empty(self) -> None:
        token_ids = [1, 2, 3]
        self.assertEqual(_token_ids_from_spans(token_ids=token_ids, spans=[(0, 2)], prefix_len=0), [])


if __name__ == "__main__":
    unittest.main()

