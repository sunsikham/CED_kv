from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectedSpan:
    start: int
    end: int
    score: float


def select_contiguous_spans(
    scores: list[float],
    span_len: int,
    budget_spans: int,
    min_distance: int,
) -> list[SelectedSpan]:
    if span_len <= 0:
        raise ValueError("span_len must be > 0")
    if budget_spans <= 0:
        return []
    if not scores:
        return []

    candidates: list[SelectedSpan] = []
    max_start = max(0, len(scores) - span_len)
    for start in range(max_start + 1):
        end = min(len(scores), start + span_len)
        avg_score = sum(scores[start:end]) / (end - start)
        candidates.append(SelectedSpan(start=start, end=end, score=avg_score))
    candidates.sort(key=lambda item: item.score, reverse=True)

    chosen: list[SelectedSpan] = []
    for cand in candidates:
        if len(chosen) >= budget_spans:
            break
        if _is_far_enough(cand=cand, chosen=chosen, min_distance=min_distance):
            chosen.append(cand)
    return chosen


def _is_far_enough(cand: SelectedSpan, chosen: list[SelectedSpan], min_distance: int) -> bool:
    for existing in chosen:
        overlap = not (cand.end <= existing.start or cand.start >= existing.end)
        if overlap:
            return False
        distance = min(abs(cand.start - existing.end), abs(existing.start - cand.end))
        if distance < min_distance:
            return False
    return True

