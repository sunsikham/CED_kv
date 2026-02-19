from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .model_hf import HFModelRuntime, encode_text


@dataclass(frozen=True)
class KVCaptureMeta:
    layer_count: int
    head_count: int
    head_dim: int
    seq_len: int
    layer_scope: str
    token_text: list[str]
    token_index_to_span: list[tuple[int, int]]
    position_index: list[int]
    past_len: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_mock_demo_kv(
    demo_text: str,
    layer_scope: str,
    layer_count: int = 16,
    head_count: int = 8,
    head_dim: int = 64,
) -> dict[str, Any]:
    tokens = demo_text.split()
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = demo_text.find(token, cursor)
        end = start + len(token)
        spans.append((start, end))
        cursor = end

    selected_layers = _select_layer_ids(layer_count=layer_count, layer_scope=layer_scope)
    meta = KVCaptureMeta(
        layer_count=len(selected_layers),
        head_count=head_count,
        head_dim=head_dim,
        seq_len=len(tokens),
        layer_scope=layer_scope,
        token_text=tokens,
        token_index_to_span=spans,
        position_index=list(range(len(tokens))),
    )
    return {
        "meta": meta.to_dict(),
        "layer_ids": selected_layers,
        "kv_payload": {"type": "mock", "seq_len": len(tokens), "layers": selected_layers},
    }


def score_query_to_demo_attention_mock(
    demo_tokens: list[str],
    answer_token: str,
    layer_scope: str,
) -> dict[str, Any]:
    scores: list[float] = []
    for token in demo_tokens:
        score = 0.05
        if answer_token in token:
            score = 1.0
        elif "->" in token:
            score = 0.30
        scores.append(score)

    return {
        "aggregation": "query_to_demo_mean_top25_layers",
        "layer_scope": layer_scope,
        "scores": scores,
    }


def capture_hf_demo_kv(
    runtime: HFModelRuntime,
    demo_text: str,
    layer_scope: str,
) -> dict[str, Any]:
    encoded = encode_text(runtime=runtime, text=demo_text)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with runtime.torch.no_grad():
        outputs = runtime.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    past = outputs.past_key_values
    if past is None:
        raise ValueError("HF model did not return past_key_values with use_cache=True")

    layer_count = len(past)
    layer_ids = _select_layer_ids(layer_count=layer_count, layer_scope=layer_scope)
    first_k = past[0][0]
    head_count = int(first_k.shape[1])
    seq_len = int(first_k.shape[2])
    head_dim = int(first_k.shape[3])

    token_ids = input_ids[0].tolist()
    tokens = runtime.tokenizer.convert_ids_to_tokens(token_ids)
    spans = _token_spans_from_offsets(runtime=runtime, text=demo_text)
    if len(spans) != len(tokens):
        spans = [(0, 0) for _ in tokens]

    meta = KVCaptureMeta(
        layer_count=len(layer_ids),
        head_count=head_count,
        head_dim=head_dim,
        seq_len=seq_len,
        layer_scope=layer_scope,
        token_text=list(tokens),
        token_index_to_span=spans,
        position_index=list(range(seq_len)),
        past_len=seq_len,
    )
    return {
        "meta": meta.to_dict(),
        "layer_ids": layer_ids,
        "kv_payload": {
            "type": "hf",
            "past_key_values": past,
            "past_len": seq_len,
            "all_layer_count": layer_count,
        },
    }


def _token_spans_from_offsets(runtime: HFModelRuntime, text: str) -> list[tuple[int, int]]:
    try:
        encoded = runtime.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping", [])
        return [(int(start), int(end)) for start, end in offsets]
    except Exception:
        return []


def _select_layer_ids(layer_count: int, layer_scope: str) -> list[int]:
    if layer_scope == "all":
        return list(range(layer_count))
    if layer_scope == "top25":
        top_n = max(1, layer_count // 4)
        return list(range(layer_count - top_n, layer_count))
    if layer_scope == "single":
        return [layer_count // 2]
    raise ValueError(f"Unsupported layer_scope: {layer_scope}")
