from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any

from .eval_phase1 import (
    _phase1_teacher_full_text,
    _phase1_teacher_prefill_text,
    _run_gate_a_hf,
)
from .metrics_phase0 import percentile
from .model_hf import (
    HFBackendUnavailable,
    HFModelRuntime,
    answer_first_token_candidate_ids,
    encode_text,
    forward_next_token_logits,
    forward_next_token_logits_with_past,
    kl_from_logits,
    load_hf_model,
    relative_kl,
)
from .runtime import now_iso
from .synthetic import generate_phase1_off_samples, generate_phase1_on_samples


@dataclass(frozen=True)
class Phase2Settings:
    model_id: str
    device: str
    runtime_backend: str
    allow_mock_fallback: bool
    torch_dtype: str
    prefix_len: int
    positioning_mode: str
    padding_mode: str
    on_num_samples: int
    off_num_samples: int
    repro_runs: int
    n_random_trials: int
    topk_tokens: int
    decoding: str
    select_mode: str
    span_len: int
    span_budget: int
    min_span_distance: int
    candidate_policy: str
    center_policy: str
    layer_scope_selection: str
    layer_scope_injection: str
    teacher_min_divergence: float
    alpha_roundtrip: float
    eps_null_floor: float
    eps_null_multiplier: float
    eps_nonzero_multiplier: float
    eps_teacher_cache_floor: float
    eps_teacher_cache_multiplier: float
    on_gain_min: float
    off_delta_p99_max: float
    delta_on_min: float
    teacher_min_on_acc: float
    rel_kl_to_teacher_max: float
    repro_acc_tol: float


def resolve_phase2_settings(
    config: dict[str, Any],
    model_id_override: str | None = None,
    device_override: str | None = None,
    max_on_samples: int | None = None,
    max_off_samples: int | None = None,
    prefix_len_override: int | None = None,
    select_mode_override: str | None = None,
    strict_hf_override: bool | None = None,
    dtype_override: str | None = None,
) -> Phase2Settings:
    phase2 = config.get("phase2", {})
    runtime = phase2.get("runtime", {})
    thresholds = phase2.get("thresholds", {})
    evaluation = phase2.get("eval", {})
    select = phase2.get("select", {})
    reporting = phase2.get("reporting", {})
    injection = phase2.get("injection", {})
    prefix = phase2.get("prefix", {})
    layer_scope = phase2.get("layer_scope", {})

    model_cfg = phase2.get("model", {})
    model_id = str(model_id_override or model_cfg.get("id") or config.get("model", {}).get("id", "unknown-model"))
    on_num_samples = int(max_on_samples or evaluation.get("on_samples", 64))
    off_num_samples = int(max_off_samples or evaluation.get("off_samples", 256))
    prefix_len = int(prefix_len_override or phase2.get("prefix_len", 32))
    if on_num_samples <= 0 or off_num_samples <= 0:
        raise ValueError("phase2 eval sample counts must be > 0")
    if prefix_len < 0:
        raise ValueError("phase2.prefix_len must be >= 0")

    allow_fallback = bool(runtime.get("allow_mock_fallback", False))
    if strict_hf_override is True:
        allow_fallback = False

    return Phase2Settings(
        model_id=model_id,
        device=str(device_override or runtime.get("device", "auto")),
        runtime_backend=str(runtime.get("backend", "hf")),
        allow_mock_fallback=allow_fallback,
        torch_dtype=str(dtype_override or runtime.get("torch_dtype", "auto")),
        prefix_len=prefix_len,
        positioning_mode=str(injection.get("positioning_mode", "compact")),
        padding_mode=str(prefix.get("padding_mode", "null_pad")),
        on_num_samples=on_num_samples,
        off_num_samples=off_num_samples,
        repro_runs=max(1, int(evaluation.get("repro_runs", 2))),
        n_random_trials=max(1, int(evaluation.get("n_random_trials", 5))),
        topk_tokens=max(1, int(reporting.get("topk_tokens", 16))),
        decoding=str(evaluation.get("decoding", "greedy")),
        select_mode=str(select_mode_override or select.get("mode", "attention_diversity_span")),
        span_len=max(1, int(select.get("span_len", 4))),
        span_budget=max(1, int(select.get("span_budget", 4))),
        min_span_distance=max(0, int(select.get("min_span_distance", 1))),
        candidate_policy=str(select.get("candidate_policy", "input_and_delimiter_only")),
        center_policy=str(select.get("center_policy", "start")),
        layer_scope_selection=str(layer_scope.get("selection", "top25")),
        layer_scope_injection=str(layer_scope.get("injection", "all")),
        teacher_min_divergence=float(thresholds.get("teacher_min_divergence", 0.02)),
        alpha_roundtrip=float(thresholds.get("alpha_roundtrip", 0.5)),
        eps_null_floor=float(thresholds.get("eps_null_floor", 1e-3)),
        eps_null_multiplier=float(thresholds.get("eps_null_multiplier", 10.0)),
        eps_nonzero_multiplier=float(thresholds.get("eps_nonzero_multiplier", 10.0)),
        eps_teacher_cache_floor=float(thresholds.get("eps_teacher_cache_floor", 1e-3)),
        eps_teacher_cache_multiplier=float(thresholds.get("eps_teacher_cache_multiplier", 10.0)),
        on_gain_min=float(thresholds.get("on_gain_min", 0.02)),
        off_delta_p99_max=float(thresholds.get("off_delta_p99_max", 0.05)),
        delta_on_min=float(thresholds.get("delta_on_min", 0.01)),
        teacher_min_on_acc=float(thresholds.get("teacher_min_on_acc", 0.2)),
        rel_kl_to_teacher_max=float(thresholds.get("rel_kl_to_teacher_max", 1.0)),
        repro_acc_tol=float(thresholds.get("repro_acc_tol", 0.02)),
    )


def run_phase2(
    config: dict[str, Any],
    seed: int,
    settings: Phase2Settings,
) -> dict[str, Any]:
    rng = random.Random(seed)
    runtime = _resolve_hf_runtime(settings=settings)

    synthetic_cfg = config.get("data", {}).get("synthetic", {})
    on_samples = generate_phase1_on_samples(config=synthetic_cfg, rng=rng, num_samples=settings.on_num_samples)
    off_samples = generate_phase1_off_samples(rng=rng, num_samples=settings.off_num_samples)

    gatea_artifacts: dict[str, Any] = {"gateA_kv_meta": [], "hf_runtime_meta": {}}
    gatea_records: list[dict[str, Any]] = []
    gatea = _run_gate_a_hf(
        on_samples=on_samples,
        off_samples=off_samples,
        settings=settings,
        runtime=runtime,
        sample_records=gatea_records,
        artifacts=gatea_artifacts,
    )

    sample_records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {
        "gateA_kv_meta": gatea_artifacts.get("gateA_kv_meta", []),
        "hf_runtime_meta": gatea_artifacts.get("hf_runtime_meta", {}),
        "phase2_selection": [],
        "phase2_prefix_meta": [],
        "phase2_off_stress_top": [],
    }
    gateb = _run_gate_b_hf(
        on_samples=on_samples,
        off_samples=off_samples,
        settings=settings,
        runtime=runtime,
        rng=rng,
        sample_records=sample_records,
        artifacts=artifacts,
    )

    run_metrics = [
        ("phase2_on_acc_base", gateb["on_acc_base"]),
        ("phase2_on_acc_teacher", gateb["on_acc_teacher"]),
        ("phase2_on_acc_student_select", gateb["on_acc_student_select"]),
        ("phase2_on_acc_student_random_mean", gateb["on_acc_student_random_mean"]),
        ("phase2_on_gain_select", gateb["on_gain_select"]),
        ("phase2_off_delta_p99_select", gateb["off_delta_p99_select"]),  # stress alias for compatibility
        ("phase2_off_delta_p99_stress", gateb["off_delta_p99_stress"]),
        ("phase2_off_delta_p99_typical", gateb["off_delta_p99_typical"]),
        ("phase2_off_delta_mean_stress", gateb["off_delta_mean_stress"]),
        ("phase2_off_delta_mean_typical", gateb["off_delta_mean_typical"]),
        ("phase2_delta_on_mean_select", gateb["delta_on_mean_select"]),
        ("phase2_rel_kl_to_teacher_select", gateb["rel_kl_to_teacher_select"]),
        ("phase2_answer_logprob_gain_mean", gateb["answer_logprob_gain_mean"]),
        ("phase2_gateA_pass", 1.0 if gatea["pass"] else 0.0),
        ("phase2_gateB_pass", 1.0 if gateb["pass"] else 0.0),
    ]

    report = {
        "phase": 2,
        "plan_version": "phase2_v2",
        "backend": "hf",
        "backend_reason": "hf runtime active",
        "device": settings.device,
        "model_id": settings.model_id,
        "decoding": settings.decoding,
        "positioning_mode": settings.positioning_mode,
        "padding_mode": settings.padding_mode,
        "delta_definition": "KL(student||base) over full-vocab logits",
        "off_eval_definitions": {
            "stress": "max_i KL(student(state_i)||base) per off sample",
            "typical": "KL(student(state_rr)||base) with round-robin state assignment",
        },
        "gates": {"A": gatea, "B": gateb},
        "sample_counts": {
            "on": len(on_samples),
            "off": len(off_samples),
            "total": len(on_samples) + len(off_samples),
        },
    }
    return {
        "sample_records": sample_records,
        "run_metrics": run_metrics,
        "report": report,
        "gate_pass": bool(gatea["pass"] and gateb["pass"]),
        "gateA_pass": bool(gatea["pass"]),
        "gateB_pass": bool(gateb["pass"]),
        "artifacts": artifacts,
    }


def _resolve_hf_runtime(settings: Phase2Settings) -> HFModelRuntime:
    if settings.runtime_backend != "hf":
        raise HFBackendUnavailable(f"phase2 requires runtime backend 'hf', got: {settings.runtime_backend}")
    if settings.allow_mock_fallback:
        raise HFBackendUnavailable("phase2 disallows mock fallback; set allow_mock_fallback=false")
    return load_hf_model(model_id=settings.model_id, device=settings.device, torch_dtype=settings.torch_dtype)


def _run_gate_b_hf(
    on_samples: list[dict[str, Any]],
    off_samples: list[dict[str, Any]],
    settings: Phase2Settings,
    runtime: HFModelRuntime,
    rng: random.Random,
    sample_records: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    if settings.positioning_mode != "compact":
        raise ValueError(f"Unsupported phase2.injection.positioning_mode: {settings.positioning_mode}")

    base_hits: list[float] = []
    teacher_hits: list[float] = []
    selected_hits: list[float] = []
    random_mean_hits: list[float] = []
    on_deltas: list[float] = []
    rel_kl_to_teacher_values: list[float] = []
    answer_logprob_gains: list[float] = []
    off_deltas_stress: list[float] = []
    off_deltas_typical: list[float] = []
    state_pool: list[dict[str, Any]] = []

    for step, sample in enumerate(on_samples):
        prefill_text = _phase1_teacher_prefill_text(sample)
        query_text = str(sample["query_text"])
        teacher_text = _phase1_teacher_full_text(sample)

        base = forward_next_token_logits(runtime=runtime, text=query_text, use_cache=True)
        teacher = forward_next_token_logits(runtime=runtime, text=teacher_text, use_cache=True)

        token_meta = _prefill_token_meta(runtime=runtime, prefill_text=prefill_text)
        attention = _attention_scores_query_to_prefill(
            runtime=runtime,
            prefill_text=prefill_text,
            query_text=query_text,
            layer_scope=settings.layer_scope_selection,
        )
        candidate_mask = _candidate_mask_for_policy(
            prefill_text=prefill_text,
            token_spans=token_meta["spans"],
            policy=settings.candidate_policy,
        )
        spans = _select_spans_with_policy(
            scores=attention["scores"],
            candidate_mask=candidate_mask,
            span_len=settings.span_len,
            span_budget=settings.span_budget,
            min_distance=settings.min_span_distance,
            center_policy=settings.center_policy,
        )
        selected_token_ids = _token_ids_from_spans(
            token_ids=token_meta["token_ids"],
            spans=spans,
            prefix_len=settings.prefix_len,
        )
        if not selected_token_ids:
            selected_token_ids = token_meta["token_ids"][: settings.prefix_len]
        compact_text = runtime.tokenizer.decode(
            selected_token_ids,
            clean_up_tokenization_spaces=False,
        )
        compact_capture = _capture_compact_prefix(runtime=runtime, compact_text=compact_text)
        student_select = forward_next_token_logits_with_past(
            runtime=runtime,
            query_text=query_text,
            past_key_values=compact_capture["past_key_values"],
            past_len=compact_capture["past_len"],
        )

        answer_candidates = answer_first_token_candidate_ids(
            runtime=runtime,
            answer_text=str(sample["answer"]),
        )
        base_hit = _answer_hit(logits=base["logits"], answer_candidates=answer_candidates)
        teacher_hit = _answer_hit(logits=teacher["logits"], answer_candidates=answer_candidates)
        selected_hit = _answer_hit(logits=student_select["logits"], answer_candidates=answer_candidates)
        random_hit_values: list[float] = []
        for _ in range(settings.n_random_trials):
            random_spans = _sample_random_spans(
                candidate_mask=candidate_mask,
                seq_len=len(attention["scores"]),
                span_len=settings.span_len,
                span_budget=settings.span_budget,
                min_distance=settings.min_span_distance,
                center_policy=settings.center_policy,
                rng=rng,
            )
            random_ids = _token_ids_from_spans(
                token_ids=token_meta["token_ids"],
                spans=random_spans,
                prefix_len=settings.prefix_len,
            )
            if not random_ids:
                random_ids = token_meta["token_ids"][: settings.prefix_len]
            random_text = runtime.tokenizer.decode(
                random_ids,
                clean_up_tokenization_spaces=False,
            )
            random_capture = _capture_compact_prefix(runtime=runtime, compact_text=random_text)
            student_random = forward_next_token_logits_with_past(
                runtime=runtime,
                query_text=query_text,
                past_key_values=random_capture["past_key_values"],
                past_len=random_capture["past_len"],
            )
            random_hit_values.append(_answer_hit(logits=student_random["logits"], answer_candidates=answer_candidates))

        delta_on = kl_from_logits(student_select["logits"], base["logits"])
        rel_kl_teacher = relative_kl(
            student_teacher_kl=kl_from_logits(student_select["logits"], teacher["logits"]),
            base_teacher_kl=kl_from_logits(base["logits"], teacher["logits"]),
        )
        answer_logprob_gain = _answer_logprob_gain(
            runtime=runtime,
            answer_candidates=answer_candidates,
            selected_logits=student_select["logits"],
            base_logits=base["logits"],
        )

        base_hits.append(base_hit)
        teacher_hits.append(teacher_hit)
        selected_hits.append(selected_hit)
        random_mean_hits.append(sum(random_hit_values) / len(random_hit_values))
        on_deltas.append(delta_on)
        rel_kl_to_teacher_values.append(rel_kl_teacher)
        answer_logprob_gains.append(answer_logprob_gain)

        sample_records.extend(
            [
                _sample_metric(
                    sample=sample,
                    metric="phase2_delta_on_select",
                    value=delta_on,
                    model_id=settings.model_id,
                    prefix_len=settings.prefix_len,
                    step=step,
                ),
                _sample_metric(
                    sample=sample,
                    metric="phase2_answer_logprob_gain",
                    value=answer_logprob_gain,
                    model_id=settings.model_id,
                    prefix_len=settings.prefix_len,
                    step=step,
                ),
            ]
        )

        artifacts["phase2_selection"].append(
            {
                "sample_id": sample["sample_id"],
                "episode_id": sample["episode_id"],
                "mode": sample["mode"],
                "selection_mode": settings.select_mode,
                "aggregation": attention["aggregation"],
                "spans": spans,
                "selected_token_count": len(selected_token_ids),
            }
        )
        artifacts["phase2_prefix_meta"].append(
            {
                "sample_id": sample["sample_id"],
                "episode_id": sample["episode_id"],
                "positioning_mode": settings.positioning_mode,
                "padding_mode": settings.padding_mode,
                "prefix_len": settings.prefix_len,
                "effective_prefix_len": len(selected_token_ids),
                "null_pad_tokens": max(0, settings.prefix_len - len(selected_token_ids)),
            }
        )

        state_pool.append(
            {
                "state_id": f"on_{step}",
                "past_key_values": compact_capture["past_key_values"],
                "past_len": compact_capture["past_len"],
                "episode_id": sample["episode_id"],
            }
        )

    if not state_pool:
        raise ValueError("phase2 requires at least one ON sample to build OFF evaluation state")

    for local_idx, sample in enumerate(off_samples):
        step = len(on_samples) + local_idx
        base = forward_next_token_logits(runtime=runtime, text=sample["query_text"], use_cache=True)
        stress_delta = -1.0
        stress_state_id = ""
        state_deltas: list[float] = []
        for state in state_pool:
            student_off = forward_next_token_logits_with_past(
                runtime=runtime,
                query_text=sample["query_text"],
                past_key_values=state["past_key_values"],
                past_len=state["past_len"],
            )
            delta = kl_from_logits(student_off["logits"], base["logits"])
            state_deltas.append(delta)
            if delta > stress_delta:
                stress_delta = delta
                stress_state_id = str(state["state_id"])

        rr_idx = local_idx % len(state_pool)
        typical_delta = state_deltas[rr_idx]
        off_deltas_stress.append(stress_delta)
        off_deltas_typical.append(typical_delta)
        artifacts["phase2_off_stress_top"].append(
            {
                "sample_id": sample["sample_id"],
                "episode_id": sample["episode_id"],
                "stress_state_id": stress_state_id,
                "stress_delta": stress_delta,
                "typical_state_id": str(state_pool[rr_idx]["state_id"]),
                "typical_delta": typical_delta,
            }
        )
        sample_records.extend(
            [
                _sample_metric(
                    sample=sample,
                    metric="phase2_delta_off_stress_select",
                    value=stress_delta,
                    model_id=settings.model_id,
                    prefix_len=settings.prefix_len,
                    step=step,
                ),
                _sample_metric(
                    sample=sample,
                    metric="phase2_delta_off_typical_select",
                    value=typical_delta,
                    model_id=settings.model_id,
                    prefix_len=settings.prefix_len,
                    step=step,
                ),
                _sample_metric(
                    sample=sample,
                    metric="phase2_delta_off_select",
                    value=stress_delta,
                    model_id=settings.model_id,
                    prefix_len=settings.prefix_len,
                    step=step,
                ),
            ]
        )

    on_acc_base = sum(base_hits) / len(base_hits)
    on_acc_teacher = sum(teacher_hits) / len(teacher_hits)
    on_acc_student_select = sum(selected_hits) / len(selected_hits)
    on_acc_student_random_mean = sum(random_mean_hits) / len(random_mean_hits)
    on_gain = on_acc_student_select - on_acc_base
    delta_on_mean = sum(on_deltas) / len(on_deltas)
    off_delta_p99_stress = percentile(off_deltas_stress, 0.99)
    off_delta_p99_typical = percentile(off_deltas_typical, 0.99)
    off_delta_mean_stress = sum(off_deltas_stress) / len(off_deltas_stress)
    off_delta_mean_typical = sum(off_deltas_typical) / len(off_deltas_typical)
    rel_kl_to_teacher_mean = sum(rel_kl_to_teacher_values) / len(rel_kl_to_teacher_values)
    answer_logprob_gain_mean = sum(answer_logprob_gains) / len(answer_logprob_gains)

    core_gate_pass = _gateb_pass(
        on_gain=on_gain,
        on_gain_min=settings.on_gain_min,
        off_delta_p99=off_delta_p99_stress,
        off_delta_p99_max=settings.off_delta_p99_max,
        delta_on_mean=delta_on_mean,
        delta_on_min=settings.delta_on_min,
        rel_kl_to_teacher=rel_kl_to_teacher_mean,
        rel_kl_to_teacher_max=settings.rel_kl_to_teacher_max,
        on_acc_select=on_acc_student_select,
        on_acc_random_mean=on_acc_student_random_mean,
    )
    on_eval_valid = on_acc_teacher >= settings.teacher_min_on_acc
    gate_pass = bool(on_eval_valid and core_gate_pass)
    if not on_eval_valid:
        failure_reason = "on_eval_invalid"
    else:
        failure_reason = _gateb_failure_reason(
            on_gain=on_gain,
            on_gain_min=settings.on_gain_min,
            off_delta_p99=off_delta_p99_stress,
            off_delta_p99_max=settings.off_delta_p99_max,
            delta_on_mean=delta_on_mean,
            delta_on_min=settings.delta_on_min,
            rel_kl_to_teacher=rel_kl_to_teacher_mean,
            rel_kl_to_teacher_max=settings.rel_kl_to_teacher_max,
            on_acc_select=on_acc_student_select,
            on_acc_random_mean=on_acc_student_random_mean,
        )
    return {
        "required": True,
        "pass": gate_pass,
        "on_acc_base": on_acc_base,
        "on_acc_teacher": on_acc_teacher,
        "on_acc_student_select": on_acc_student_select,
        "on_acc_student_random_mean": on_acc_student_random_mean,
        "on_gain_select": on_gain,
        "on_gain_min": settings.on_gain_min,
        "on_eval_valid": on_eval_valid,
        "teacher_min_on_acc": settings.teacher_min_on_acc,
        "failure_reason": failure_reason,
        "off_delta_p99_select": off_delta_p99_stress,
        "off_delta_p99_stress": off_delta_p99_stress,
        "off_delta_p99_typical": off_delta_p99_typical,
        "off_delta_mean_stress": off_delta_mean_stress,
        "off_delta_mean_typical": off_delta_mean_typical,
        "off_delta_p99_max": settings.off_delta_p99_max,
        "delta_on_mean_select": delta_on_mean,
        "delta_on_min": settings.delta_on_min,
        "rel_kl_to_teacher_select": rel_kl_to_teacher_mean,
        "rel_kl_to_teacher_max": settings.rel_kl_to_teacher_max,
        "answer_logprob_gain_mean": answer_logprob_gain_mean,
        "n_random_trials": settings.n_random_trials,
        "positioning_mode": settings.positioning_mode,
        "padding_mode": settings.padding_mode,
        "candidate_policy": settings.candidate_policy,
        "center_policy": settings.center_policy,
    }


def _capture_compact_prefix(runtime: HFModelRuntime, compact_text: str) -> dict[str, Any]:
    encoded = encode_text(runtime=runtime, text=compact_text)
    with runtime.torch.no_grad():
        outputs = runtime.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=True,
            return_dict=True,
        )
    if outputs.past_key_values is None:
        raise ValueError("HF model did not return past_key_values for compact prefix")
    return {
        "past_key_values": outputs.past_key_values,
        "past_len": int(encoded["input_ids"].shape[1]),
    }


def _prefill_token_meta(runtime: HFModelRuntime, prefill_text: str) -> dict[str, Any]:
    encoded = runtime.tokenizer(
        prefill_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(token) for token in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded.get("offset_mapping", [])]
    if len(offsets) != len(token_ids):
        offsets = [(0, 0) for _ in token_ids]
    return {
        "token_ids": token_ids,
        "spans": offsets,
    }


def _attention_scores_query_to_prefill(
    runtime: HFModelRuntime,
    prefill_text: str,
    query_text: str,
    layer_scope: str,
) -> dict[str, Any]:
    full_text = prefill_text + query_text
    full = encode_text(runtime=runtime, text=full_text)
    prefill = encode_text(runtime=runtime, text=prefill_text)
    prefill_len = int(prefill["input_ids"].shape[1])
    seq_len = int(full["input_ids"].shape[1])
    query_len = max(0, seq_len - prefill_len)

    previous_impl = _get_model_attn_impl(runtime=runtime)
    switched = _maybe_set_model_attn_impl(runtime=runtime, attn_impl="eager")
    try:
        with runtime.torch.no_grad():
            outputs = runtime.model(
                input_ids=full["input_ids"],
                attention_mask=full["attention_mask"],
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        if switched and previous_impl is not None and previous_impl != "eager":
            _maybe_set_model_attn_impl(runtime=runtime, attn_impl=previous_impl)
    attentions = outputs.attentions or ()
    if not attentions or prefill_len == 0 or query_len == 0:
        return {
            "scores": [0.0 for _ in range(prefill_len)],
            "aggregation": "query_to_prefill_mean_attention",
        }

    layer_ids = _select_layer_ids(layer_count=len(attentions), layer_scope=layer_scope)
    agg = runtime.torch.zeros((prefill_len,), dtype=runtime.torch.float32)
    for layer_id in layer_ids:
        layer_attn = attentions[layer_id][0].to(runtime.torch.float32)
        query_to_prefill = layer_attn[:, prefill_len : prefill_len + query_len, :prefill_len]
        if query_to_prefill.numel() == 0:
            continue
        token_scores = query_to_prefill.mean(dim=(0, 1))
        agg += token_scores.detach().cpu()
    if layer_ids:
        agg /= float(len(layer_ids))

    return {
        "scores": [float(value) for value in agg.tolist()],
        "aggregation": "query_to_prefill_mean_attention",
    }


def _get_model_attn_impl(runtime: HFModelRuntime) -> str | None:
    model = runtime.model
    config = getattr(model, "config", None)
    if config is None:
        return None
    for name in ("_attn_implementation", "attn_implementation"):
        value = getattr(config, name, None)
        if isinstance(value, str) and value:
            return value
    return None


def _maybe_set_model_attn_impl(runtime: HFModelRuntime, attn_impl: str) -> bool:
    model = runtime.model
    setter = getattr(model, "set_attn_implementation", None)
    if setter is None:
        return False
    try:
        setter(attn_impl)
        return True
    except Exception:
        return False


def _select_layer_ids(layer_count: int, layer_scope: str) -> list[int]:
    if layer_scope == "all":
        return list(range(layer_count))
    if layer_scope == "top25":
        top_n = max(1, layer_count // 4)
        return list(range(layer_count - top_n, layer_count))
    raise ValueError(f"Unsupported layer scope: {layer_scope}")


def _candidate_mask_for_policy(
    prefill_text: str,
    token_spans: list[tuple[int, int]],
    policy: str,
) -> list[bool]:
    if policy == "all":
        return [True for _ in token_spans]
    if policy != "input_and_delimiter_only":
        raise ValueError(f"Unsupported phase2.select.candidate_policy: {policy}")
    value_ranges = _value_ranges(prefill_text=prefill_text)
    mask: list[bool] = []
    for start, end in token_spans:
        is_value = any(not (end <= left or start >= right) for left, right in value_ranges)
        mask.append(not is_value)
    return mask


def _value_ranges(prefill_text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"->\s*([^\n]+)", prefill_text):
        start, end = match.span(1)
        ranges.append((start, end))
    return ranges


def _select_spans_with_policy(
    scores: list[float],
    candidate_mask: list[bool],
    span_len: int,
    span_budget: int,
    min_distance: int,
    center_policy: str,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int, float]] = []
    max_start = max(0, len(scores) - span_len)
    for start in range(max_start + 1):
        end = min(len(scores), start + span_len)
        center_idx = _center_index(start=start, end=end, policy=center_policy)
        if center_idx < 0 or center_idx >= len(candidate_mask):
            continue
        if not candidate_mask[center_idx]:
            continue
        avg_score = sum(scores[start:end]) / max(1, end - start)
        candidates.append((start, end, avg_score))
    candidates.sort(key=lambda item: item[2], reverse=True)

    chosen: list[tuple[int, int]] = []
    for start, end, _ in candidates:
        if len(chosen) >= span_budget:
            break
        if _is_far_enough(start=start, end=end, chosen=chosen, min_distance=min_distance):
            chosen.append((start, end))
    return chosen


def _sample_random_spans(
    candidate_mask: list[bool],
    seq_len: int,
    span_len: int,
    span_budget: int,
    min_distance: int,
    center_policy: str,
    rng: random.Random,
) -> list[tuple[int, int]]:
    starts: list[int] = []
    max_start = max(0, seq_len - span_len)
    for start in range(max_start + 1):
        end = min(seq_len, start + span_len)
        center_idx = _center_index(start=start, end=end, policy=center_policy)
        if 0 <= center_idx < len(candidate_mask) and candidate_mask[center_idx]:
            starts.append(start)
    rng.shuffle(starts)
    chosen: list[tuple[int, int]] = []
    for start in starts:
        if len(chosen) >= span_budget:
            break
        end = min(seq_len, start + span_len)
        if _is_far_enough(start=start, end=end, chosen=chosen, min_distance=min_distance):
            chosen.append((start, end))
    return chosen


def _center_index(start: int, end: int, policy: str) -> int:
    if policy == "start":
        return start
    if policy == "center":
        return start + max(0, (end - start - 1) // 2)
    raise ValueError(f"Unsupported center policy: {policy}")


def _is_far_enough(start: int, end: int, chosen: list[tuple[int, int]], min_distance: int) -> bool:
    for chosen_start, chosen_end in chosen:
        overlap = not (end <= chosen_start or start >= chosen_end)
        if overlap:
            return False
        distance = min(abs(start - chosen_end), abs(chosen_start - end))
        if distance < min_distance:
            return False
    return True


def _token_ids_from_spans(token_ids: list[int], spans: list[tuple[int, int]], prefix_len: int) -> list[int]:
    if prefix_len <= 0 or not token_ids:
        return []
    selected: list[int] = []
    for start, end in sorted(spans, key=lambda item: item[0]):
        left = max(0, start)
        right = min(len(token_ids), end)
        if right <= left:
            continue
        selected.extend(token_ids[left:right])
        if len(selected) >= prefix_len:
            break
    return selected[:prefix_len]


def _answer_hit(logits: Any, answer_candidates: list[int]) -> float:
    if not answer_candidates:
        return 0.0
    pred = int(logits.argmax(dim=-1).item())
    return float(pred in set(answer_candidates))


def _answer_logprob_gain(
    runtime: HFModelRuntime,
    answer_candidates: list[int],
    selected_logits: Any,
    base_logits: Any,
) -> float:
    if not answer_candidates:
        return 0.0
    indices = runtime.torch.tensor(answer_candidates, dtype=runtime.torch.long)
    logp_selected_all = selected_logits.detach().to("cpu").float().log_softmax(dim=-1)[0]
    logp_base_all = base_logits.detach().to("cpu").float().log_softmax(dim=-1)[0]
    logp_selected = runtime.torch.logsumexp(logp_selected_all[indices], dim=0).item()
    logp_base = runtime.torch.logsumexp(logp_base_all[indices], dim=0).item()
    return float(logp_selected - logp_base)


def _gateb_pass(
    on_gain: float,
    on_gain_min: float,
    off_delta_p99: float,
    off_delta_p99_max: float,
    delta_on_mean: float,
    delta_on_min: float,
    rel_kl_to_teacher: float,
    rel_kl_to_teacher_max: float,
    on_acc_select: float,
    on_acc_random_mean: float,
) -> bool:
    return (
        on_gain >= on_gain_min
        and off_delta_p99 <= off_delta_p99_max
        and delta_on_mean >= delta_on_min
        and rel_kl_to_teacher <= rel_kl_to_teacher_max
        and on_acc_select >= on_acc_random_mean
    )


def _gateb_failure_reason(
    on_gain: float,
    on_gain_min: float,
    off_delta_p99: float,
    off_delta_p99_max: float,
    delta_on_mean: float,
    delta_on_min: float,
    rel_kl_to_teacher: float,
    rel_kl_to_teacher_max: float,
    on_acc_select: float,
    on_acc_random_mean: float,
) -> str:
    if on_gain < on_gain_min:
        return "on_gain_below_min"
    if off_delta_p99 > off_delta_p99_max:
        return "off_delta_stress_p99_above_max"
    if delta_on_mean < delta_on_min:
        return "delta_on_below_min"
    if rel_kl_to_teacher > rel_kl_to_teacher_max:
        return "rel_kl_to_teacher_above_max"
    if on_acc_select < on_acc_random_mean:
        return "select_below_random_mean"
    return "ok"


def _sample_metric(
    sample: dict[str, Any],
    metric: str,
    value: float,
    model_id: str,
    prefix_len: int,
    step: int,
) -> dict[str, Any]:
    prompt = sample.get("query_text") or sample.get("teacher_text") or ""
    return {
        "record_type": "sample",
        "step": step,
        "split": sample["mode"],
        "mode": sample["mode"],
        "metric": metric,
        "value": float(value),
        "timestamp": now_iso(),
        "model_id": model_id,
        "prompt_len": len(str(prompt).split()),
        "prefix_len": prefix_len,
        "episode_id": sample["episode_id"],
        "sample_id": sample["sample_id"],
    }
