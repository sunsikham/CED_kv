from __future__ import annotations

from typing import Any


def force_all_layer_ids(total_layers: int) -> list[int]:
    if total_layers < 0:
        raise ValueError("total_layers must be >= 0")
    return list(range(total_layers))


def build_null_prefix(prefix_len: int, layer_ids: list[int]) -> dict[str, Any]:
    return {
        "type": "null",
        "prefix_len": max(0, prefix_len),
        "layer_ids": layer_ids,
        "spans": [],
    }


def build_full_demo_prefix(prefix_len: int, layer_ids: list[int], seq_len: int) -> dict[str, Any]:
    effective_len = min(max(0, prefix_len), max(0, seq_len))
    return {
        "type": "full_demo",
        "prefix_len": effective_len,
        "layer_ids": layer_ids,
        "spans": [(0, effective_len)],
    }


def build_selected_span_prefix(
    prefix_len: int,
    layer_ids: list[int],
    selected_spans: list[tuple[int, int]],
) -> dict[str, Any]:
    spans = _truncate_spans_to_prefix(selected_spans=selected_spans, prefix_len=prefix_len)
    return {
        "type": "selected_span",
        "prefix_len": prefix_len,
        "layer_ids": layer_ids,
        "spans": spans,
    }


def build_random_span_prefix(
    prefix_len: int,
    layer_ids: list[int],
    seq_len: int,
    span_len: int,
) -> dict[str, Any]:
    if prefix_len <= 0 or seq_len <= 0:
        spans: list[tuple[int, int]] = []
    else:
        end = min(seq_len, span_len)
        spans = [(0, end)]
    return {
        "type": "random_span",
        "prefix_len": prefix_len,
        "layer_ids": layer_ids,
        "spans": spans,
    }


def _truncate_spans_to_prefix(selected_spans: list[tuple[int, int]], prefix_len: int) -> list[tuple[int, int]]:
    if prefix_len <= 0:
        return []
    total = 0
    result: list[tuple[int, int]] = []
    for start, end in selected_spans:
        if end <= start:
            continue
        span_size = end - start
        remain = prefix_len - total
        if remain <= 0:
            break
        if span_size <= remain:
            result.append((start, end))
            total += span_size
        else:
            result.append((start, start + remain))
            total += remain
            break
    return result
