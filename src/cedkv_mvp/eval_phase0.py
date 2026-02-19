from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics_phase0 import (
    kl_student_to_base_union_other,
    percentile,
    to_topk_sparse,
)
from .predictors_phase0 import (
    BASE_DEFAULT_TOKEN,
    GLOBAL_STEER_TOKEN,
    build_phase0_predictors,
)
from .runtime import now_iso
from .synthetic import generate_phase0_off_samples, generate_phase0_on_samples


@dataclass(frozen=True)
class Phase0Settings:
    stub_mode: str
    threshold_profile: str
    on_num_samples: int
    off_num_samples: int
    off_delta_p99_max: float
    topk_tokens: int


def resolve_phase0_settings(
    config: dict[str, Any],
    stub_mode_override: str | None = None,
    max_on_samples: int | None = None,
    max_off_samples: int | None = None,
) -> Phase0Settings:
    phase0 = config.get("phase0", {})
    on_eval = phase0.get("on_eval", {})
    off_eval = phase0.get("off_eval", {})
    reporting = phase0.get("reporting", {})
    thresholds = phase0.get("thresholds", {})

    configured_stub_mode = phase0.get("stub_mode", "null")
    if configured_stub_mode is None:
        configured_stub_mode = "null"
    stub_mode = stub_mode_override or str(configured_stub_mode)
    on_num_samples = int(max_on_samples or on_eval.get("num_samples", 64))
    off_num_samples = int(max_off_samples or off_eval.get("num_samples", 256))

    if on_num_samples <= 0:
        raise ValueError("phase0.on_eval.num_samples must be > 0")
    if off_num_samples <= 0:
        raise ValueError("phase0.off_eval.num_samples must be > 0")

    return Phase0Settings(
        stub_mode=stub_mode,
        threshold_profile=str(phase0.get("threshold_profile", "phase0_stub")),
        on_num_samples=on_num_samples,
        off_num_samples=off_num_samples,
        off_delta_p99_max=float(thresholds.get("off_delta_p99_max", 0.05)),
        topk_tokens=int(reporting.get("topk_tokens", 8)),
    )


def _argmax_token(distribution: dict[str, float]) -> str:
    return max(distribution.items(), key=lambda item: item[1])[0]


def _build_vocab(on_samples: list[dict[str, str]]) -> list[str]:
    answer_tokens = sorted({sample["answer"] for sample in on_samples if sample["answer"]})
    extras = [BASE_DEFAULT_TOKEN, GLOBAL_STEER_TOKEN, "OTHER_A", "OTHER_B", "OTHER_C"]
    ordered: list[str] = []
    for token in answer_tokens + extras:
        if token not in ordered:
            ordered.append(token)
    return ordered


def run_phase0(
    config: dict[str, Any],
    seed: int,
    model_id: str,
    settings: Phase0Settings,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    synthetic_cfg = config.get("data", {}).get("synthetic", {})
    on_samples = generate_phase0_on_samples(
        config=synthetic_cfg,
        rng=rng,
        num_samples=settings.on_num_samples,
    )
    off_samples = generate_phase0_off_samples(rng=rng, num_samples=settings.off_num_samples)

    vocab = _build_vocab(on_samples=on_samples)
    predictors = build_phase0_predictors(stub_mode=settings.stub_mode, vocab=vocab)

    sample_records: list[dict[str, Any]] = []
    on_base_correct: list[float] = []
    on_teacher_correct: list[float] = []
    on_student_correct: list[float] = []

    for step, sample in enumerate(on_samples):
        base = predictors.base(sample)
        teacher = predictors.teacher(sample)
        student = predictors.student(sample)
        answer = sample["answer"]
        base_hit = float(_argmax_token(base) == answer)
        teacher_hit = float(_argmax_token(teacher) == answer)
        student_hit = float(_argmax_token(student) == answer)
        on_base_correct.append(base_hit)
        on_teacher_correct.append(teacher_hit)
        on_student_correct.append(student_hit)

        sample_records.extend(
            [
                {
                    "record_type": "sample",
                    "step": step,
                    "split": "on",
                    "mode": "on",
                    "metric": "on_exact_match_base",
                    "value": base_hit,
                    "timestamp": now_iso(),
                    "model_id": model_id,
                    "prompt_len": len(sample["prompt"].split()),
                    "prefix_len": 0,
                    "episode_id": sample["episode_id"],
                    "sample_id": sample["sample_id"],
                },
                {
                    "record_type": "sample",
                    "step": step,
                    "split": "on",
                    "mode": "on",
                    "metric": "on_exact_match_teacher",
                    "value": teacher_hit,
                    "timestamp": now_iso(),
                    "model_id": model_id,
                    "prompt_len": len(sample["prompt"].split()),
                    "prefix_len": 0,
                    "episode_id": sample["episode_id"],
                    "sample_id": sample["sample_id"],
                },
                {
                    "record_type": "sample",
                    "step": step,
                    "split": "on",
                    "mode": "on",
                    "metric": "on_exact_match_student",
                    "value": student_hit,
                    "timestamp": now_iso(),
                    "model_id": model_id,
                    "prompt_len": len(sample["prompt"].split()),
                    "prefix_len": 0,
                    "episode_id": sample["episode_id"],
                    "sample_id": sample["sample_id"],
                },
            ]
        )

    off_deltas: list[float] = []
    for offset, sample in enumerate(off_samples, start=len(on_samples)):
        base = predictors.base(sample)
        student = predictors.student(sample)
        base_sparse = to_topk_sparse(dist=base, top_k=settings.topk_tokens)
        student_sparse = to_topk_sparse(dist=student, top_k=settings.topk_tokens)
        delta = kl_student_to_base_union_other(
            student_sparse=student_sparse,
            base_sparse=base_sparse,
        )
        off_deltas.append(delta)
        sample_records.append(
            {
                "record_type": "sample",
                "step": offset,
                "split": "off",
                "mode": "off",
                "metric": "off_delta_kl_student_to_base",
                "value": delta,
                "timestamp": now_iso(),
                "model_id": model_id,
                "prompt_len": len(sample["prompt"].split()),
                "prefix_len": 0,
                "episode_id": sample["episode_id"],
                "sample_id": sample["sample_id"],
            }
        )

    on_base_acc = sum(on_base_correct) / len(on_base_correct)
    on_teacher_acc = sum(on_teacher_correct) / len(on_teacher_correct)
    on_student_acc = sum(on_student_correct) / len(on_student_correct)

    off_delta_mean = sum(off_deltas) / len(off_deltas)
    off_delta_p95 = percentile(off_deltas, 0.95)
    off_delta_p99 = percentile(off_deltas, 0.99)

    gate_pass = off_delta_p99 <= settings.off_delta_p99_max
    run_metrics = [
        ("on_exact_match_acc_base", on_base_acc),
        ("on_exact_match_acc_teacher", on_teacher_acc),
        ("on_exact_match_acc_student", on_student_acc),
        ("off_delta_kl_student_to_base_mean", off_delta_mean),
        ("off_delta_kl_student_to_base_p95", off_delta_p95),
        ("off_delta_kl_student_to_base_p99", off_delta_p99),
        ("phase0_gate_pass", 1.0 if gate_pass else 0.0),
    ]

    report = {
        "phase": 0,
        "stub_mode": settings.stub_mode,
        "threshold_profile": settings.threshold_profile,
        "delta_definition": "KL(student||base) over union(top-k) + OTHER",
        "on_metrics": {
            "on_exact_match_acc_base": on_base_acc,
            "on_exact_match_acc_teacher": on_teacher_acc,
            "on_exact_match_acc_student": on_student_acc,
        },
        "off_metrics": {
            "off_delta_kl_student_to_base_mean": off_delta_mean,
            "off_delta_kl_student_to_base_p95": off_delta_p95,
            "off_delta_kl_student_to_base_p99": off_delta_p99,
        },
        "gates": {
            "pass": gate_pass,
            "off_delta_p99_threshold": settings.off_delta_p99_max,
            "off_delta_p99_observed": off_delta_p99,
        },
        "sample_counts": {
            "on": len(on_samples),
            "off": len(off_samples),
            "total": len(on_samples) + len(off_samples),
        },
    }

    return {
        "run_metrics": run_metrics,
        "sample_records": sample_records,
        "report": report,
        "gate_pass": gate_pass,
    }
