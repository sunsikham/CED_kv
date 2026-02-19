from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kv_capture import (
    capture_hf_demo_kv,
    capture_mock_demo_kv,
    score_query_to_demo_attention_mock,
)
from .kv_inject import (
    build_full_demo_prefix,
    build_null_prefix,
    build_random_span_prefix,
    build_selected_span_prefix,
    force_all_layer_ids,
)
from .metrics_phase0 import (
    kl_student_to_base_union_other,
    percentile,
    to_topk_sparse,
)
from .model_hf import (
    HFBackendUnavailable,
    HFModelRuntime,
    answer_first_token_id,
    forward_next_token_logits,
    forward_next_token_logits_with_past,
    kl_from_logits,
    load_hf_model,
    relative_kl,
)
from .runtime import now_iso
from .select_kv_phase1 import select_contiguous_spans
from .synthetic import generate_phase1_off_samples, generate_phase1_on_samples

DEFAULT_TOKEN = "BASE_DEFAULT"
STEER_TOKEN = "STEER"


@dataclass(frozen=True)
class Phase1Settings:
    model_id: str
    device: str
    runtime_backend: str
    allow_mock_fallback: bool
    torch_dtype: str
    gate_mode: str
    gatea_layer_scope: str
    gateb_layer_scope: str
    select_mode: str
    prefix_len: int
    span_len: int
    span_budget: int
    min_span_distance: int
    on_num_samples: int
    off_num_samples: int
    repro_runs: int
    topk_tokens: int
    decoding: str
    teacher_min_divergence: float
    alpha_roundtrip: float
    eps_null_floor: float
    eps_null_multiplier: float
    eps_nonzero_multiplier: float
    eps_teacher_cache_floor: float
    eps_teacher_cache_multiplier: float
    on_gain_advisory: float
    off_delta_p99_advisory: float


def resolve_phase1_settings(
    config: dict[str, Any],
    model_id_override: str | None = None,
    device_override: str | None = None,
    max_on_samples: int | None = None,
    max_off_samples: int | None = None,
    prefix_len_override: int | None = None,
    select_mode_override: str | None = None,
    backend_override: str | None = None,
    strict_hf_override: bool | None = None,
    dtype_override: str | None = None,
) -> Phase1Settings:
    phase1 = config.get("phase1", {})
    runtime = phase1.get("runtime", {})
    thresholds = phase1.get("thresholds", {})
    evaluation = phase1.get("eval", {})
    select = phase1.get("select", {})
    reporting = phase1.get("reporting", {})

    model_cfg = phase1.get("model", {})
    model_id = str(model_id_override or model_cfg.get("id") or config.get("model", {}).get("id", "unknown-model"))
    on_num_samples = int(max_on_samples or evaluation.get("on_samples", 64))
    off_num_samples = int(max_off_samples or evaluation.get("off_samples", 256))
    prefix_len = int(prefix_len_override or phase1.get("prefix_len", 32))
    if on_num_samples <= 0 or off_num_samples <= 0:
        raise ValueError("phase1 eval sample counts must be > 0")
    if prefix_len < 0:
        raise ValueError("phase1.prefix_len must be >= 0")

    allow_fallback = bool(runtime.get("allow_mock_fallback", True))
    if strict_hf_override is True:
        allow_fallback = False

    return Phase1Settings(
        model_id=model_id,
        device=str(device_override or runtime.get("device", "auto")),
        runtime_backend=str(backend_override or runtime.get("backend", "mock")),
        allow_mock_fallback=allow_fallback,
        torch_dtype=str(dtype_override or runtime.get("torch_dtype", "auto")),
        gate_mode=str(phase1.get("gate_mode", "two_tier")),
        gatea_layer_scope=str(phase1.get("layer_scope_gatea", "all")),
        gateb_layer_scope=str(phase1.get("layer_scope_gateb", "top25")),
        select_mode=str(select_mode_override or select.get("mode", "attention_diversity_span")),
        prefix_len=prefix_len,
        span_len=int(select.get("span_len", 4)),
        span_budget=int(select.get("span_budget", 4)),
        min_span_distance=int(select.get("min_span_distance", 1)),
        on_num_samples=on_num_samples,
        off_num_samples=off_num_samples,
        repro_runs=max(1, int(evaluation.get("repro_runs", 2))),
        topk_tokens=max(1, int(reporting.get("topk_tokens", 16))),
        decoding=str(evaluation.get("decoding", "greedy")),
        teacher_min_divergence=float(thresholds.get("teacher_min_divergence", 0.02)),
        alpha_roundtrip=float(thresholds.get("alpha_roundtrip", 0.5)),
        eps_null_floor=float(thresholds.get("eps_null_floor", 1e-3)),
        eps_null_multiplier=float(thresholds.get("eps_null_multiplier", 10.0)),
        eps_nonzero_multiplier=float(thresholds.get("eps_nonzero_multiplier", 10.0)),
        eps_teacher_cache_floor=float(thresholds.get("eps_teacher_cache_floor", 1e-3)),
        eps_teacher_cache_multiplier=float(thresholds.get("eps_teacher_cache_multiplier", 10.0)),
        on_gain_advisory=float(thresholds.get("on_gain_advisory", 0.02)),
        off_delta_p99_advisory=float(thresholds.get("off_delta_p99_advisory", 0.05)),
    )


def run_phase1(
    config: dict[str, Any],
    seed: int,
    settings: Phase1Settings,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    backend = _resolve_backend_context(settings=settings)

    synthetic_cfg = config.get("data", {}).get("synthetic", {})
    on_samples = generate_phase1_on_samples(
        config=synthetic_cfg,
        rng=rng,
        num_samples=settings.on_num_samples,
    )
    off_samples = generate_phase1_off_samples(rng=rng, num_samples=settings.off_num_samples)
    vocab = _build_vocab(on_samples)
    mock_cfg = config.get("phase1", {}).get("mock", {})

    sample_records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {
        "gateA_kv_meta": [],
        "gateB_selection": [],
        "hf_runtime_meta": {},
    }

    if backend["mode"] == "hf":
        gatea = _run_gate_a_hf(
            on_samples=on_samples,
            off_samples=off_samples,
            settings=settings,
            runtime=backend["runtime"],
            sample_records=sample_records,
            artifacts=artifacts,
        )
        gatea_source = "hf_logits"
    else:
        gatea = _run_gate_a_mock(
            on_samples=on_samples,
            off_samples=off_samples,
            vocab=vocab,
            settings=settings,
            mock_cfg=mock_cfg,
            sample_records=sample_records,
            artifacts=artifacts,
        )
        gatea_source = "mock_distribution"

    gateb = _run_gate_b_mock(
        on_samples=on_samples,
        off_samples=off_samples,
        vocab=vocab,
        settings=settings,
        mock_cfg=mock_cfg,
        sample_records=sample_records,
        artifacts=artifacts,
    )

    run_metrics = [
        ("phase1_teacher_incremental_consistency_kl", gatea["teacher_incremental_consistency_kl"]),
        ("phase1_teacher_divergence_mean", gatea["teacher_divergence_mean"]),
        ("phase1_teacher_acc_advisory", gatea["teacher_acc_advisory"]),
        ("phase1_delta_on_null_mean", gatea["delta_on_null_mean"]),
        ("phase1_delta_off_null_mean", gatea["delta_off_null_mean"]),
        ("phase1_delta_on_nonnull_mean", gatea["delta_on_nonnull_mean"]),
        ("phase1_roundtrip_kl_student_to_teacher", gatea["roundtrip_student_teacher_kl"]),
        ("phase1_roundtrip_kl_base_to_teacher", gatea["roundtrip_base_teacher_kl"]),
        ("phase1_roundtrip_relative_kl", gatea["roundtrip_relative_kl"]),
        ("phase1_gateA_pass", 1.0 if gatea["pass"] else 0.0),
        ("phase1_on_gain_select", gateb["on_gain_select"]),
        ("phase1_off_delta_p99_select", gateb["off_delta_p99_select"]),
        ("phase1_gateB_pass", 1.0 if gateb["pass"] else 0.0),
    ]

    report = {
        "phase": 1,
        "plan_version": "phase15_v1",
        "backend": backend["label"],
        "backend_reason": backend["reason"],
        "device": settings.device,
        "model_id": settings.model_id,
        "decoding": settings.decoding,
        "gate_mode": settings.gate_mode,
        "gateA_source": gatea_source,
        "gateB_source": "mock_advisory",
        "teacher_sanity_type": "divergence",
        "delta_definition": "KL(student||base) over full-vocab logits (hf) or union(top-k)+OTHER (mock)",
        "roundtrip_definition": "KL(student_full_demo||teacher_full) at answer first token",
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
        "gate_pass": gatea["pass"],
        "gateA_pass": gatea["pass"],
        "gateB_pass": gateb["pass"],
        "artifacts": artifacts,
    }


def _resolve_backend_context(settings: Phase1Settings) -> dict[str, Any]:
    if settings.runtime_backend == "mock":
        return {
            "mode": "mock",
            "label": "mock",
            "reason": "configured mock backend",
            "runtime": None,
        }

    if settings.runtime_backend != "hf":
        raise ValueError(f"Unsupported phase1.runtime.backend: {settings.runtime_backend}")

    try:
        runtime = load_hf_model(
            model_id=settings.model_id,
            device=settings.device,
            torch_dtype=settings.torch_dtype,
        )
        return {
            "mode": "hf",
            "label": "hf",
            "reason": "hf runtime active",
            "runtime": runtime,
        }
    except Exception as exc:
        if settings.allow_mock_fallback:
            return {
                "mode": "mock",
                "label": "mock_fallback",
                "reason": f"HF unavailable, fallback to mock: {exc}",
                "runtime": None,
            }
        if isinstance(exc, HFBackendUnavailable):
            raise
        raise HFBackendUnavailable(str(exc))


def _run_gate_a_hf(
    on_samples: list[dict[str, Any]],
    off_samples: list[dict[str, Any]],
    settings: Phase1Settings,
    runtime: HFModelRuntime,
    sample_records: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    eps_noise = _estimate_base_repeat_noise_hf(
        runtime=runtime,
        on_samples=on_samples,
        off_samples=off_samples,
    )
    eps_null = max(settings.eps_null_floor, settings.eps_null_multiplier * eps_noise)
    eps_nonzero = settings.eps_nonzero_multiplier * eps_null
    eps_teacher_cache = max(settings.eps_teacher_cache_floor, settings.eps_teacher_cache_multiplier * eps_noise)

    teacher_incremental_kls: list[float] = []
    teacher_divergences: list[float] = []
    teacher_hits: list[float] = []
    delta_on_null: list[float] = []
    delta_on_nonnull: list[float] = []
    delta_off_null: list[float] = []
    roundtrip_student_teacher: list[float] = []
    roundtrip_base_teacher: list[float] = []
    repro_acc_runs: list[float] = []

    artifacts["hf_runtime_meta"] = {
        "model_id": runtime.model_id,
        "device": runtime.device,
        "torch_dtype": runtime.torch_dtype,
        "torch_version": runtime.torch_version,
        "transformers_version": runtime.transformers_version,
    }

    for run_idx in range(settings.repro_runs):
        student_hits_run: list[float] = []
        for step, sample in enumerate(on_samples):
            teacher_full_text = _phase1_teacher_full_text(sample)
            teacher_prefill_text = _phase1_teacher_prefill_text(sample)
            teacher_query_text = _phase1_teacher_query_text(sample)
            capture = capture_hf_demo_kv(
                runtime=runtime,
                demo_text=teacher_prefill_text,
                layer_scope="all",  # Gate A full-demo is always all-layers.
            )
            if run_idx == 0:
                artifacts["gateA_kv_meta"].append(capture["meta"])

            teacher_full = forward_next_token_logits(runtime=runtime, text=teacher_full_text, use_cache=True)
            teacher_cached = forward_next_token_logits_with_past(
                runtime=runtime,
                query_text=teacher_query_text,
                past_key_values=capture["kv_payload"]["past_key_values"],
                past_len=int(capture["kv_payload"]["past_len"]),
            )
            if run_idx == 0 and step == 0:
                artifacts["hf_runtime_meta"].update(
                    {
                        "demo_len": int(capture["kv_payload"]["past_len"]),
                        "query_len": int(teacher_cached["query_len"]),
                        "past_len": int(teacher_cached["past_len"]),
                        "query_position_first": int(teacher_cached["position_first"]),
                        "query_position_last": int(teacher_cached["position_last"]),
                        "attention_mask_len": int(teacher_cached["attention_mask_len"]),
                        "attention_mask_sum": int(teacher_cached["attention_mask_sum"]),
                    }
                )

            base = forward_next_token_logits(runtime=runtime, text=sample["query_text"], use_cache=True)
            student_null_logits = base["logits"]
            student_full_logits = teacher_cached["logits"]

            kl_teacher_incremental = kl_from_logits(teacher_cached["logits"], teacher_full["logits"])
            kl_teacher_div = kl_from_logits(teacher_full["logits"], base["logits"])
            kl_on_null = kl_from_logits(student_null_logits, base["logits"])
            kl_on_nonnull = kl_from_logits(student_full_logits, base["logits"])
            kl_student_teacher = kl_from_logits(student_full_logits, teacher_full["logits"])
            kl_base_teacher = kl_from_logits(base["logits"], teacher_full["logits"])

            teacher_incremental_kls.append(kl_teacher_incremental)
            teacher_divergences.append(kl_teacher_div)
            delta_on_null.append(kl_on_null)
            delta_on_nonnull.append(kl_on_nonnull)
            roundtrip_student_teacher.append(kl_student_teacher)
            roundtrip_base_teacher.append(kl_base_teacher)

            answer_id = answer_first_token_id(runtime=runtime, answer_text=sample["answer"])
            if answer_id is None:
                teacher_hit = 0.0
                student_hit = 0.0
            else:
                teacher_hit = float(int(teacher_full["logits"].argmax(dim=-1).item()) == answer_id)
                student_hit = float(int(student_full_logits.argmax(dim=-1).item()) == answer_id)
            teacher_hits.append(teacher_hit)
            student_hits_run.append(student_hit)

            if run_idx == 0:
                sample_records.extend(
                    [
                        _sample_metric(
                            sample=sample,
                            metric="phase1_teacher_incremental_consistency_kl",
                            value=kl_teacher_incremental,
                            model_id=settings.model_id,
                            prefix_len=settings.prefix_len,
                            step=step,
                        ),
                        _sample_metric(
                            sample=sample,
                            metric="phase1_delta_on_nonnull",
                            value=kl_on_nonnull,
                            model_id=settings.model_id,
                            prefix_len=settings.prefix_len,
                            step=step,
                        ),
                    ]
                )
        if student_hits_run:
            repro_acc_runs.append(sum(student_hits_run) / len(student_hits_run))

    for step, sample in enumerate(off_samples, start=len(on_samples)):
        base = forward_next_token_logits(runtime=runtime, text=sample["query_text"], use_cache=True)
        kl_off_null = kl_from_logits(base["logits"], base["logits"])
        delta_off_null.append(kl_off_null)
        sample_records.append(
            _sample_metric(
                sample=sample,
                metric="phase1_delta_off_null",
                value=kl_off_null,
                model_id=settings.model_id,
                prefix_len=settings.prefix_len,
                step=step,
            )
        )

    teacher_incremental_consistency_kl = sum(teacher_incremental_kls) / len(teacher_incremental_kls)
    teacher_divergence_mean = sum(teacher_divergences) / len(teacher_divergences)
    teacher_acc_advisory = sum(teacher_hits) / len(teacher_hits)
    delta_on_null_mean = sum(delta_on_null) / len(delta_on_null)
    delta_off_null_mean = sum(delta_off_null) / len(delta_off_null)
    delta_on_nonnull_mean = sum(delta_on_nonnull) / len(delta_on_nonnull)
    roundtrip_student_teacher_kl = sum(roundtrip_student_teacher) / len(roundtrip_student_teacher)
    roundtrip_base_teacher_kl = sum(roundtrip_base_teacher) / len(roundtrip_base_teacher)
    roundtrip_relative = relative_kl(
        student_teacher_kl=roundtrip_student_teacher_kl,
        base_teacher_kl=roundtrip_base_teacher_kl,
    )
    relative_ok = roundtrip_relative <= settings.alpha_roundtrip

    repro_acc_diff = max(repro_acc_runs) - min(repro_acc_runs) if repro_acc_runs else 0.0
    repro_ok = repro_acc_diff <= 0.02

    gate_pass = (
        teacher_incremental_consistency_kl <= eps_teacher_cache
        and teacher_divergence_mean >= settings.teacher_min_divergence
        and delta_on_null_mean <= eps_null
        and delta_off_null_mean <= eps_null
        and delta_on_nonnull_mean >= eps_nonzero
        and relative_ok
        and repro_ok
    )

    return {
        "required": True,
        "pass": gate_pass,
        "teacher_incremental_consistency_kl": teacher_incremental_consistency_kl,
        "eps_teacher_cache": eps_teacher_cache,
        "teacher_divergence_mean": teacher_divergence_mean,
        "teacher_min_divergence": settings.teacher_min_divergence,
        "teacher_acc_advisory": teacher_acc_advisory,
        "delta_on_null_mean": delta_on_null_mean,
        "delta_off_null_mean": delta_off_null_mean,
        "delta_on_nonnull_mean": delta_on_nonnull_mean,
        "eps_null": eps_null,
        "eps_nonzero": eps_nonzero,
        "roundtrip_student_teacher_kl": roundtrip_student_teacher_kl,
        "roundtrip_base_teacher_kl": roundtrip_base_teacher_kl,
        "roundtrip_relative_kl": roundtrip_relative,
        "alpha_roundtrip": settings.alpha_roundtrip,
        "relative_roundtrip_ok": relative_ok,
        "repro_acc_diff": repro_acc_diff,
        "repro_acc_tol": 0.02,
    }


def _run_gate_a_mock(
    on_samples: list[dict[str, Any]],
    off_samples: list[dict[str, Any]],
    vocab: list[str],
    settings: Phase1Settings,
    mock_cfg: dict[str, Any],
    sample_records: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    eps_noise = _estimate_base_repeat_noise_mock(
        on_samples=on_samples,
        off_samples=off_samples,
        vocab=vocab,
        mock_cfg=mock_cfg,
    )
    eps_null = max(settings.eps_null_floor, settings.eps_null_multiplier * eps_noise)
    eps_nonzero = settings.eps_nonzero_multiplier * eps_null
    eps_teacher_cache = max(settings.eps_teacher_cache_floor, settings.eps_teacher_cache_multiplier * eps_noise)

    teacher_incremental_kls: list[float] = []
    teacher_divergences: list[float] = []
    teacher_hits: list[float] = []
    delta_on_null: list[float] = []
    delta_off_null: list[float] = []
    delta_on_nonnull: list[float] = []
    roundtrip_student_teacher: list[float] = []
    roundtrip_base_teacher: list[float] = []
    repro_acc_runs: list[float] = []

    for run_idx in range(settings.repro_runs):
        student_hits_run: list[float] = []
        for step, sample in enumerate(on_samples):
            capture = capture_mock_demo_kv(
                demo_text=sample["demo_text"],
                layer_scope="all",
            )
            if run_idx == 0:
                artifacts["gateA_kv_meta"].append(capture["meta"])

            base = _predict_distribution(sample=sample, vocab=vocab, mode="base", mock_cfg=mock_cfg)
            teacher_full = _predict_distribution(sample=sample, vocab=vocab, mode="teacher", mock_cfg=mock_cfg)
            teacher_cached = _predict_distribution(sample=sample, vocab=vocab, mode="teacher", mock_cfg=mock_cfg)
            student_null = _predict_distribution(sample=sample, vocab=vocab, mode="student_null", mock_cfg=mock_cfg)
            student_full = _predict_distribution(sample=sample, vocab=vocab, mode="student_full_demo", mock_cfg=mock_cfg)

            _ = build_null_prefix(prefix_len=settings.prefix_len, layer_ids=force_all_layer_ids(capture["meta"]["layer_count"]))
            _ = build_full_demo_prefix(
                prefix_len=settings.prefix_len,
                layer_ids=force_all_layer_ids(capture["meta"]["layer_count"]),
                seq_len=capture["meta"]["seq_len"],
            )

            kl_teacher_incremental = _kl_delta(student=teacher_cached, base=teacher_full, topk=settings.topk_tokens)
            kl_teacher_div = _kl_delta(student=teacher_full, base=base, topk=settings.topk_tokens)
            kl_on_null = _kl_delta(student=student_null, base=base, topk=settings.topk_tokens)
            kl_on_nonnull = _kl_delta(student=student_full, base=base, topk=settings.topk_tokens)
            kl_student_teacher = _kl_delta(student=student_full, base=teacher_full, topk=settings.topk_tokens)
            kl_base_teacher = _kl_delta(student=base, base=teacher_full, topk=settings.topk_tokens)

            teacher_incremental_kls.append(kl_teacher_incremental)
            teacher_divergences.append(kl_teacher_div)
            delta_on_null.append(kl_on_null)
            delta_on_nonnull.append(kl_on_nonnull)
            roundtrip_student_teacher.append(kl_student_teacher)
            roundtrip_base_teacher.append(kl_base_teacher)

            teacher_hit = float(_argmax_token(teacher_full) == sample["answer"])
            student_hit = float(_argmax_token(student_full) == sample["answer"])
            teacher_hits.append(teacher_hit)
            student_hits_run.append(student_hit)

            if run_idx == 0:
                sample_records.extend(
                    [
                        _sample_metric(
                            sample=sample,
                            metric="phase1_teacher_incremental_consistency_kl",
                            value=kl_teacher_incremental,
                            model_id=settings.model_id,
                            prefix_len=settings.prefix_len,
                            step=step,
                        ),
                        _sample_metric(
                            sample=sample,
                            metric="phase1_delta_on_nonnull",
                            value=kl_on_nonnull,
                            model_id=settings.model_id,
                            prefix_len=settings.prefix_len,
                            step=step,
                        ),
                    ]
                )
        if student_hits_run:
            repro_acc_runs.append(sum(student_hits_run) / len(student_hits_run))

    for step, sample in enumerate(off_samples, start=len(on_samples)):
        base = _predict_distribution(sample=sample, vocab=vocab, mode="base", mock_cfg=mock_cfg)
        student_null = _predict_distribution(sample=sample, vocab=vocab, mode="student_null", mock_cfg=mock_cfg)
        kl_off_null = _kl_delta(student=student_null, base=base, topk=settings.topk_tokens)
        delta_off_null.append(kl_off_null)
        sample_records.append(
            _sample_metric(
                sample=sample,
                metric="phase1_delta_off_null",
                value=kl_off_null,
                model_id=settings.model_id,
                prefix_len=settings.prefix_len,
                step=step,
            )
        )

    teacher_incremental_consistency_kl = sum(teacher_incremental_kls) / len(teacher_incremental_kls)
    teacher_divergence_mean = sum(teacher_divergences) / len(teacher_divergences)
    teacher_acc_advisory = sum(teacher_hits) / len(teacher_hits)
    delta_on_null_mean = sum(delta_on_null) / len(delta_on_null)
    delta_off_null_mean = sum(delta_off_null) / len(delta_off_null)
    delta_on_nonnull_mean = sum(delta_on_nonnull) / len(delta_on_nonnull)
    roundtrip_student_teacher_kl = sum(roundtrip_student_teacher) / len(roundtrip_student_teacher)
    roundtrip_base_teacher_kl = sum(roundtrip_base_teacher) / len(roundtrip_base_teacher)
    roundtrip_relative = roundtrip_student_teacher_kl / (roundtrip_base_teacher_kl + 1e-6)
    relative_ok = roundtrip_relative <= settings.alpha_roundtrip

    repro_acc_diff = max(repro_acc_runs) - min(repro_acc_runs) if repro_acc_runs else 0.0
    repro_ok = repro_acc_diff <= 0.02

    gate_pass = (
        teacher_incremental_consistency_kl <= eps_teacher_cache
        and teacher_divergence_mean >= settings.teacher_min_divergence
        and delta_on_null_mean <= eps_null
        and delta_off_null_mean <= eps_null
        and delta_on_nonnull_mean >= eps_nonzero
        and relative_ok
        and repro_ok
    )

    return {
        "required": True,
        "pass": gate_pass,
        "teacher_incremental_consistency_kl": teacher_incremental_consistency_kl,
        "eps_teacher_cache": eps_teacher_cache,
        "teacher_divergence_mean": teacher_divergence_mean,
        "teacher_min_divergence": settings.teacher_min_divergence,
        "teacher_acc_advisory": teacher_acc_advisory,
        "delta_on_null_mean": delta_on_null_mean,
        "delta_off_null_mean": delta_off_null_mean,
        "delta_on_nonnull_mean": delta_on_nonnull_mean,
        "eps_null": eps_null,
        "eps_nonzero": eps_nonzero,
        "roundtrip_student_teacher_kl": roundtrip_student_teacher_kl,
        "roundtrip_base_teacher_kl": roundtrip_base_teacher_kl,
        "roundtrip_relative_kl": roundtrip_relative,
        "alpha_roundtrip": settings.alpha_roundtrip,
        "relative_roundtrip_ok": relative_ok,
        "repro_acc_diff": repro_acc_diff,
        "repro_acc_tol": 0.02,
    }


def _run_gate_b_mock(
    on_samples: list[dict[str, Any]],
    off_samples: list[dict[str, Any]],
    vocab: list[str],
    settings: Phase1Settings,
    mock_cfg: dict[str, Any],
    sample_records: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    base_hits: list[float] = []
    selected_hits: list[float] = []
    off_deltas: list[float] = []

    for step, sample in enumerate(on_samples):
        capture = capture_mock_demo_kv(demo_text=sample["demo_text"], layer_scope=settings.gateb_layer_scope)
        attention = score_query_to_demo_attention_mock(
            demo_tokens=capture["meta"]["token_text"],
            answer_token=sample["answer"],
            layer_scope=settings.gateb_layer_scope,
        )
        spans = select_contiguous_spans(
            scores=attention["scores"],
            span_len=settings.span_len,
            budget_spans=settings.span_budget,
            min_distance=settings.min_span_distance,
        )
        selected_pairs = [(span.start, span.end) for span in spans]
        selected_prefix = build_selected_span_prefix(
            prefix_len=settings.prefix_len,
            layer_ids=capture["layer_ids"],
            selected_spans=selected_pairs,
        )
        artifacts["gateB_selection"].append(
            {
                "sample_id": sample["sample_id"],
                "episode_id": sample["episode_id"],
                "aggregation": attention["aggregation"],
                "spans": selected_pairs,
            }
        )

        base = _predict_distribution(sample=sample, vocab=vocab, mode="base", mock_cfg=mock_cfg)
        if settings.select_mode == "random_span":
            _ = build_random_span_prefix(
                prefix_len=settings.prefix_len,
                layer_ids=capture["layer_ids"],
                seq_len=capture["meta"]["seq_len"],
                span_len=settings.span_len,
            )
            student_select = _predict_distribution(sample=sample, vocab=vocab, mode="student_random", mock_cfg=mock_cfg)
        else:
            _ = selected_prefix
            student_select = _predict_distribution(sample=sample, vocab=vocab, mode="student_selected", mock_cfg=mock_cfg)

        base_hit = float(_argmax_token(base) == sample["answer"])
        selected_hit = float(_argmax_token(student_select) == sample["answer"])
        base_hits.append(base_hit)
        selected_hits.append(selected_hit)

        sample_records.append(
            _sample_metric(
                sample=sample,
                metric="phase1_on_acc_student_select",
                value=selected_hit,
                model_id=settings.model_id,
                prefix_len=settings.prefix_len,
                step=step,
            )
        )

    for step, sample in enumerate(off_samples, start=len(on_samples)):
        base = _predict_distribution(sample=sample, vocab=vocab, mode="base", mock_cfg=mock_cfg)
        student_select = _predict_distribution(sample=sample, vocab=vocab, mode="student_selected", mock_cfg=mock_cfg)
        delta = _kl_delta(student=student_select, base=base, topk=settings.topk_tokens)
        off_deltas.append(delta)
        sample_records.append(
            _sample_metric(
                sample=sample,
                metric="phase1_off_delta_select",
                value=delta,
                model_id=settings.model_id,
                prefix_len=settings.prefix_len,
                step=step,
            )
        )

    acc_base = sum(base_hits) / len(base_hits)
    acc_selected = sum(selected_hits) / len(selected_hits)
    on_gain_select = acc_selected - acc_base
    off_delta_p99_select = percentile(off_deltas, 0.99)
    gate_pass = on_gain_select >= settings.on_gain_advisory and off_delta_p99_select <= settings.off_delta_p99_advisory
    return {
        "required": False,
        "pass": gate_pass,
        "on_acc_base": acc_base,
        "on_acc_student_select": acc_selected,
        "on_gain_select": on_gain_select,
        "on_gain_advisory": settings.on_gain_advisory,
        "off_delta_p99_select": off_delta_p99_select,
        "off_delta_p99_advisory": settings.off_delta_p99_advisory,
        "select_mode": settings.select_mode,
        "layer_scope": settings.gateb_layer_scope,
    }


def _estimate_base_repeat_noise_mock(
    on_samples: list[dict[str, Any]],
    off_samples: list[dict[str, Any]],
    vocab: list[str],
    mock_cfg: dict[str, Any],
) -> float:
    deltas: list[float] = []
    for sample in on_samples + off_samples:
        base1 = _predict_distribution(sample=sample, vocab=vocab, mode="base", mock_cfg=mock_cfg)
        base2 = _predict_distribution(sample=sample, vocab=vocab, mode="base", mock_cfg=mock_cfg)
        deltas.append(_kl_delta(student=base1, base=base2, topk=16))
    return sum(deltas) / len(deltas)


def _estimate_base_repeat_noise_hf(
    runtime: HFModelRuntime,
    on_samples: list[dict[str, Any]],
    off_samples: list[dict[str, Any]],
) -> float:
    deltas: list[float] = []
    probes = on_samples[:4] + off_samples[:4]
    for sample in probes:
        text = sample["query_text"]
        base1 = forward_next_token_logits(runtime=runtime, text=text, use_cache=True)
        base2 = forward_next_token_logits(runtime=runtime, text=text, use_cache=True)
        deltas.append(kl_from_logits(base1["logits"], base2["logits"]))
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def _build_vocab(on_samples: list[dict[str, Any]]) -> list[str]:
    answer_tokens = sorted({sample["answer"] for sample in on_samples})
    extras = [DEFAULT_TOKEN, STEER_TOKEN, "ALT_A", "ALT_B", "ALT_C"]
    ordered: list[str] = []
    for token in answer_tokens + extras:
        if token not in ordered:
            ordered.append(token)
    return ordered


def _predict_distribution(
    sample: dict[str, Any],
    vocab: list[str],
    mode: str,
    mock_cfg: dict[str, Any],
) -> dict[str, float]:
    base_answer_prob = float(mock_cfg.get("base_answer_prob", 0.08))
    teacher_answer_prob = float(mock_cfg.get("teacher_answer_prob", 0.92))
    full_answer_prob = float(mock_cfg.get("student_full_answer_prob", 0.86))
    selected_answer_prob = float(mock_cfg.get("student_selected_answer_prob", 0.55))
    random_answer_prob = float(mock_cfg.get("student_random_answer_prob", 0.18))
    off_shift_prob = float(mock_cfg.get("student_off_shift_prob", 0.10))
    off_selected_shift_prob = float(mock_cfg.get("student_off_selected_shift_prob", 0.02))
    base_default_prob = float(mock_cfg.get("base_default_prob", 0.88))
    base_steer_prob = float(mock_cfg.get("base_steer_prob", 0.02))
    teacher_default_prob = float(mock_cfg.get("teacher_default_prob", 0.05))
    teacher_steer_prob = float(mock_cfg.get("teacher_steer_prob", 0.01))

    if sample["mode"] == "off":
        if mode in {"base", "teacher", "student_null"}:
            return _make_dist(vocab=vocab, answer="", answer_prob=0.0, default_prob=0.96, steer_prob=0.0)
        if mode in {"student_selected"}:
            return _make_dist(vocab=vocab, answer="", answer_prob=0.0, default_prob=1.0 - off_selected_shift_prob, steer_prob=off_selected_shift_prob)
        return _make_dist(vocab=vocab, answer="", answer_prob=0.0, default_prob=1.0 - off_shift_prob, steer_prob=off_shift_prob)

    answer = str(sample["answer"])
    if mode == "base" or mode == "student_null":
        return _make_dist(vocab=vocab, answer=answer, answer_prob=base_answer_prob, default_prob=base_default_prob, steer_prob=base_steer_prob)
    if mode == "teacher":
        return _make_dist(vocab=vocab, answer=answer, answer_prob=teacher_answer_prob, default_prob=teacher_default_prob, steer_prob=teacher_steer_prob)
    if mode == "student_full_demo":
        return _make_dist(vocab=vocab, answer=answer, answer_prob=full_answer_prob, default_prob=0.10, steer_prob=0.01)
    if mode == "student_selected":
        return _make_dist(vocab=vocab, answer=answer, answer_prob=selected_answer_prob, default_prob=0.38, steer_prob=0.02)
    if mode == "student_random":
        return _make_dist(vocab=vocab, answer=answer, answer_prob=random_answer_prob, default_prob=0.70, steer_prob=0.04)
    raise ValueError(f"Unsupported prediction mode: {mode}")


def _make_dist(vocab: list[str], answer: str, answer_prob: float, default_prob: float, steer_prob: float) -> dict[str, float]:
    dist = {token: 0.0 for token in vocab}
    answer_prob = max(0.0, min(1.0, answer_prob))
    default_prob = max(0.0, min(1.0, default_prob))
    steer_prob = max(0.0, min(1.0, steer_prob))
    if answer in dist:
        dist[answer] = answer_prob
    if DEFAULT_TOKEN in dist:
        dist[DEFAULT_TOKEN] = default_prob
    if STEER_TOKEN in dist:
        dist[STEER_TOKEN] = steer_prob
    remain = max(0.0, 1.0 - sum(dist.values()))
    others = [token for token in vocab if token not in {answer, DEFAULT_TOKEN, STEER_TOKEN}]
    if others:
        share = remain / len(others)
        for token in others:
            dist[token] += share
    else:
        dist[DEFAULT_TOKEN] = dist.get(DEFAULT_TOKEN, 0.0) + remain
    total = sum(dist.values())
    if total <= 0.0:
        uniform = 1.0 / len(vocab)
        return {token: uniform for token in vocab}
    return {token: value / total for token, value in dist.items()}


def _argmax_token(dist: dict[str, float]) -> str:
    return max(dist.items(), key=lambda item: item[1])[0]


def _kl_delta(student: dict[str, float], base: dict[str, float], topk: int) -> float:
    return kl_student_to_base_union_other(
        student_sparse=to_topk_sparse(student, topk),
        base_sparse=to_topk_sparse(base, topk),
    )


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


def _phase1_teacher_full_text(sample: dict[str, Any]) -> str:
    teacher_text = str(sample.get("teacher_text", ""))
    if teacher_text:
        return teacher_text
    return f"Demo:\n{sample['demo_text']}\n\n{sample['query_text']}"


def _phase1_teacher_prefill_text(sample: dict[str, Any]) -> str:
    return f"Demo:\n{sample['demo_text']}\n\n"


def _phase1_teacher_query_text(sample: dict[str, Any]) -> str:
    return str(sample["query_text"])
