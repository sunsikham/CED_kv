from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import random
from typing import Any

from .eval_phase1 import _phase1_teacher_full_text, _run_gate_a_hf
from .eval_phase2 import _capture_compact_prefix, _sample_metric
from .metrics_phase0 import percentile
from .model_hf import (
    HFBackendUnavailable,
    HFModelRuntime,
    answer_first_token_candidate_ids,
    forward_next_token_logits,
    forward_next_token_logits_with_past,
    kl_from_logits,
    load_hf_model,
    relative_kl,
)
from .synthetic import generate_phase1_off_samples, generate_phase1_on_samples


_FIXED_STRESS_PROMPTS = (
    "Rewrite this sentence in a formal tone.",
    "Summarize this paragraph in one sentence.",
    "Classify sentiment as positive or negative.",
    "Translate this sentence to Korean.",
    "List three concise title options.",
)


@dataclass(frozen=True)
class Phase3Settings:
    model_id: str
    device: str
    runtime_backend: str
    allow_mock_fallback: bool
    torch_dtype: str
    prefix_len: int
    on_num_samples: int
    off_num_samples: int
    repro_runs: int
    decoding: str
    topk_tokens: int
    span_budget: int
    layer_scope_mixture: str
    layer_scope_injection: str
    mix_mode: str
    k_anchor_policy: str
    atom_types: tuple[str, ...]
    delimiter_mass_cap: float
    delimiter_mass_penalty: float
    delimiter_apply_stage: str
    anti_steer_prior: str
    anti_steer_strength: float
    off_loss_type: str
    cvar_tail_fraction: float
    off_train_source: str
    off_constraint_source: str
    policy_mode: str
    policy_mode_effective: str
    ci_forced_prod: bool
    debug_allow_nonfixed_constraint_sources: bool
    off_hard_pool_size: int
    train_steps: int
    train_lr: float
    train_grad_clip: float
    constraint_warn_enabled: bool
    constraint_warn_patience_eval_ticks: int
    constraint_warn_margin: float
    constraint_eval_every: int
    topk_schedule: tuple[int, ...]
    topk_milestones: tuple[float, ...]
    lambda_on: float
    lambda_off: float
    lambda_str: float
    lambda_off_adaptive: bool
    dual_metric_source: str
    dual_eta: float
    dual_ema_beta: float
    dual_update_every: int
    dual_lambda_min: float
    dual_lambda_max: float
    dual_delta_clip: float
    off_fixed_stress_size: int
    off_fixed_stress_seed: int
    off_fixed_stress_sampling: str
    off_fixed_stress_definition: str
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
    on_gain_drop_max: float
    off_tail_improve_min: float
    phase2_baseline_on_gain: float
    phase2_baseline_off_p99: float
    milestone_explore_enabled: bool
    milestone_explore_flag: str
    milestone_explore_max_runs: int
    run_index: int
    reporting_atom_type_mass: bool


@dataclass(frozen=True)
class FixedStressEvalSpec:
    size: int
    seed: int
    sampling: str
    definition: str


@dataclass
class _EpisodeTrainState:
    sample: dict[str, Any]
    atoms: list[dict[str, str]]
    atom_types: list[str]
    warm_scores: list[float]
    warm_anchor_ids: list[int]
    logits: Any


def resolve_phase3_settings(
    config: dict[str, Any],
    model_id_override: str | None = None,
    device_override: str | None = None,
    max_on_samples: int | None = None,
    max_off_samples: int | None = None,
    prefix_len_override: int | None = None,
    strict_hf_override: bool | None = None,
    dtype_override: str | None = None,
    injection_scope_override: str | None = None,
    mix_mode_override: str | None = None,
    off_loss_type_override: str | None = None,
    cvar_tail_fraction_override: float | None = None,
    topk_schedule_override: str | None = None,
    lambda_off_adaptive_override: bool | None = None,
    explore_override: bool | None = None,
) -> Phase3Settings:
    phase2 = config.get("phase2", {})
    phase3 = config.get("phase3", {})
    runtime = phase3.get("runtime", {})
    thresholds = phase3.get("thresholds", {})
    evaluation = phase3.get("eval", {})
    train = phase3.get("train", {})
    loss = phase3.get("loss", {})
    off_loss_cfg = loss.get("off", loss.get(False, {})) if isinstance(loss, dict) else {}
    mix = phase3.get("mix", {})
    candidate = phase3.get("candidate", {})
    delimiter = candidate.get("delimiter", {})
    layer_scope = phase3.get("layer_scope", {})
    warm_start = phase3.get("warm_start", {})
    fixed_stress = evaluation.get("off_fixed_stress", {})
    reporting = phase3.get("reporting", {})
    milestone = phase3.get("milestone", {}).get("explore", {})
    dual = train.get("lambda_off", {}).get("dual", {})
    constraint_warn = train.get("constraint_warn", {})
    phase2_thresholds = phase2.get("thresholds", {})
    phase2_eval = phase2.get("eval", {})
    phase2_select = phase2.get("select", {})

    model_cfg = phase3.get("model", {})
    model_id = str(model_id_override or model_cfg.get("id") or config.get("model", {}).get("id", "unknown-model"))
    on_num_samples = int(max_on_samples or evaluation.get("on_samples", phase2_eval.get("on_samples", 64)))
    off_num_samples = int(max_off_samples or evaluation.get("off_samples", phase2_eval.get("off_samples", 256)))
    prefix_len = int(prefix_len_override or phase3.get("prefix_len", phase2.get("prefix_len", 32)))
    if on_num_samples <= 0 or off_num_samples <= 0:
        raise ValueError("phase3 eval sample counts must be > 0")
    if prefix_len <= 0:
        raise ValueError("phase3.prefix_len must be > 0")

    allow_fallback = bool(runtime.get("allow_mock_fallback", False))
    if strict_hf_override is True:
        allow_fallback = False

    topk_schedule = _parse_int_schedule(topk_schedule_override or train.get("topk_schedule", [32, 16, 8, 4]))
    topk_milestones = _parse_float_schedule(train.get("topk_milestones", [0.0, 0.25, 0.6, 0.85]))
    if len(topk_schedule) != len(topk_milestones):
        raise ValueError("phase3.train.topk_schedule and topk_milestones must have same length")

    atom_types_raw = candidate.get("atom_types", ["entry", "key_only", "value_only", "delimiter_only"])
    atom_types = tuple(str(item) for item in atom_types_raw)
    if not atom_types:
        raise ValueError("phase3.candidate.atom_types must not be empty")
    if off_loss_cfg.get("type", loss.get("off_type", "cvar")) != "cvar" and (
        off_loss_type_override or off_loss_cfg.get("type", loss.get("off_type", "cvar"))
    ) != "cvar":
        raise ValueError("phase3 currently supports only loss.off.type=cvar")

    if cvar_tail_fraction_override is not None:
        cvar_tail_fraction = float(cvar_tail_fraction_override)
    else:
        cvar_tail_fraction = float(off_loss_cfg.get("cvar_tail_fraction", loss.get("cvar_tail_fraction", 0.2)))
    if not (0.0 < cvar_tail_fraction <= 1.0):
        raise ValueError("phase3.loss.off.cvar_tail_fraction must be in (0,1]")

    off_train_source = str(off_loss_cfg.get("train_source", "hard_pool"))
    off_constraint_source = str(off_loss_cfg.get("constraint_source", "fixed_stress_eval"))
    dual_metric_source = str(dual.get("metric_source", "fixed_stress_eval"))
    if off_train_source not in {"hard_pool", "fixed_stress_eval"}:
        raise ValueError(f"Unsupported phase3.loss.off.train_source: {off_train_source}")
    for source_name, source_value in (
        ("phase3.loss.off.constraint_source", off_constraint_source),
        ("phase3.train.lambda_off.dual.metric_source", dual_metric_source),
    ):
        if source_value not in {"fixed_stress_eval", "hard_pool"}:
            raise ValueError(f"Unsupported {source_name}: {source_value}")

    policy_mode = str(runtime.get("policy_mode", "prod"))
    if policy_mode not in {"prod", "debug"}:
        raise ValueError(f"Unsupported phase3.runtime.policy_mode: {policy_mode}")
    debug_allow_nonfixed_constraint_sources = bool(runtime.get("debug_allow_nonfixed_constraint_sources", False))
    ci_forced_prod = os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}
    policy_mode_effective = "prod" if ci_forced_prod else policy_mode
    nonfixed_requested = (
        off_constraint_source != "fixed_stress_eval"
        or dual_metric_source != "fixed_stress_eval"
    )
    if policy_mode_effective == "prod" and nonfixed_requested:
        raise ValueError(
            "phase3 runtime policy_mode=prod requires fixed_stress_eval for "
            "loss.off.constraint_source and train.lambda_off.dual.metric_source"
        )
    if (
        policy_mode_effective == "debug"
        and nonfixed_requested
        and not debug_allow_nonfixed_constraint_sources
    ):
        raise ValueError(
            "phase3 runtime policy_mode=debug requires "
            "runtime.debug_allow_nonfixed_constraint_sources=true for non-fixed sources"
        )

    constraint_warn_enabled = bool(constraint_warn.get("enabled", True))
    constraint_warn_patience_eval_ticks = max(1, int(constraint_warn.get("patience_eval_ticks", 3)))
    constraint_warn_margin = max(0.0, float(constraint_warn.get("margin", 0.0)))
    constraint_eval_every = max(1, int(constraint_warn.get("eval_every", 1)))

    return Phase3Settings(
        model_id=model_id,
        device=str(device_override or runtime.get("device", "auto")),
        runtime_backend=str(runtime.get("backend", "hf")),
        allow_mock_fallback=allow_fallback,
        torch_dtype=str(dtype_override or runtime.get("torch_dtype", "auto")),
        prefix_len=prefix_len,
        on_num_samples=on_num_samples,
        off_num_samples=off_num_samples,
        repro_runs=max(1, int(evaluation.get("repro_runs", 2))),
        decoding=str(evaluation.get("decoding", "greedy")),
        topk_tokens=max(1, int(reporting.get("topk_tokens", 16))),
        span_budget=max(1, int(train.get("slot_count", phase2_select.get("span_budget", 4)))),
        layer_scope_mixture=str(layer_scope.get("mixture", "top25")),
        layer_scope_injection=str(injection_scope_override or layer_scope.get("injection", "top25")),
        mix_mode=str(mix_mode_override or mix.get("mode", "v_only")),
        k_anchor_policy=str(mix.get("k_anchor", {}).get("policy", "warm_start_fixed")),
        atom_types=atom_types,
        delimiter_mass_cap=max(0.0, min(1.0, float(delimiter.get("mass_cap", 0.15)))),
        delimiter_mass_penalty=max(0.0, min(1.0, float(delimiter.get("mass_penalty", 0.05)))),
        delimiter_apply_stage=str(delimiter.get("apply_stage", "post_topk_renorm_per_slot")),
        anti_steer_prior=str(warm_start.get("anti_steer_prior", "proxy_prefix_mass")),
        anti_steer_strength=max(0.0, float(warm_start.get("anti_steer_strength", 0.2))),
        off_loss_type=str(off_loss_type_override or off_loss_cfg.get("type", loss.get("off_type", "cvar"))),
        cvar_tail_fraction=cvar_tail_fraction,
        off_train_source=off_train_source,
        off_constraint_source=off_constraint_source,
        policy_mode=policy_mode,
        policy_mode_effective=policy_mode_effective,
        ci_forced_prod=ci_forced_prod,
        debug_allow_nonfixed_constraint_sources=debug_allow_nonfixed_constraint_sources,
        off_hard_pool_size=max(1, int(off_loss_cfg.get("hard_pool_size", 64))),
        train_steps=max(1, int(train.get("steps", 80))),
        train_lr=float(train.get("lr", 5e-2)),
        train_grad_clip=float(train.get("grad_clip", 1.0)),
        constraint_warn_enabled=constraint_warn_enabled,
        constraint_warn_patience_eval_ticks=constraint_warn_patience_eval_ticks,
        constraint_warn_margin=constraint_warn_margin,
        constraint_eval_every=constraint_eval_every,
        topk_schedule=topk_schedule,
        topk_milestones=topk_milestones,
        lambda_on=float(loss.get("lambda_on", 1.0)),
        lambda_off=float(loss.get("lambda_off", 2.0)),
        lambda_str=float(loss.get("lambda_str", 0.1)),
        lambda_off_adaptive=bool(
            lambda_off_adaptive_override
            if lambda_off_adaptive_override is not None
            else train.get("lambda_off", {}).get("adaptive", True)
        ),
        dual_metric_source=dual_metric_source,
        dual_eta=float(dual.get("lr", 0.05)),
        dual_ema_beta=float(dual.get("ema_beta", 0.9)),
        dual_update_every=max(1, int(dual.get("update_every", 5))),
        dual_lambda_min=float(dual.get("lambda_min", 0.0)),
        dual_lambda_max=float(dual.get("lambda_max", 10.0)),
        dual_delta_clip=float(dual.get("delta_clip", 0.1)),
        off_fixed_stress_size=max(1, int(fixed_stress.get("size", 2048))),
        off_fixed_stress_seed=int(fixed_stress.get("seed", 3407)),
        off_fixed_stress_sampling=str(fixed_stress.get("sampling", "once_per_experiment")),
        off_fixed_stress_definition=str(fixed_stress.get("definition", "phase2_stress_compatible")),
        teacher_min_divergence=float(thresholds.get("teacher_min_divergence", phase2_thresholds.get("teacher_min_divergence", 0.02))),
        alpha_roundtrip=float(thresholds.get("alpha_roundtrip", phase2_thresholds.get("alpha_roundtrip", 0.5))),
        eps_null_floor=float(thresholds.get("eps_null_floor", phase2_thresholds.get("eps_null_floor", 1e-3))),
        eps_null_multiplier=float(thresholds.get("eps_null_multiplier", phase2_thresholds.get("eps_null_multiplier", 10.0))),
        eps_nonzero_multiplier=float(thresholds.get("eps_nonzero_multiplier", phase2_thresholds.get("eps_nonzero_multiplier", 10.0))),
        eps_teacher_cache_floor=float(
            thresholds.get("eps_teacher_cache_floor", phase2_thresholds.get("eps_teacher_cache_floor", 1e-3))
        ),
        eps_teacher_cache_multiplier=float(
            thresholds.get("eps_teacher_cache_multiplier", phase2_thresholds.get("eps_teacher_cache_multiplier", 10.0))
        ),
        on_gain_min=float(thresholds.get("on_gain_min", phase2_thresholds.get("on_gain_min", 0.02))),
        off_delta_p99_max=float(thresholds.get("off_delta_p99_max", phase2_thresholds.get("off_delta_p99_max", 0.05))),
        delta_on_min=float(thresholds.get("delta_on_min", phase2_thresholds.get("delta_on_min", 0.01))),
        teacher_min_on_acc=float(thresholds.get("teacher_min_on_acc", phase2_thresholds.get("teacher_min_on_acc", 0.2))),
        rel_kl_to_teacher_max=float(thresholds.get("rel_kl_to_teacher_max", phase2_thresholds.get("rel_kl_to_teacher_max", 1.0))),
        on_gain_drop_max=float(milestone.get("on_gain_drop_max", 0.01)),
        off_tail_improve_min=float(milestone.get("off_tail_improve_min", 0.2)),
        phase2_baseline_on_gain=float(milestone.get("phase2_baseline_on_gain", phase2_thresholds.get("on_gain_min", 0.02))),
        phase2_baseline_off_p99=float(milestone.get("phase2_baseline_off_p99", 11.0)),
        milestone_explore_enabled=bool(milestone.get("enabled", True) if explore_override is None else explore_override),
        milestone_explore_flag=str(milestone.get("flag", "phase3_explore")),
        milestone_explore_max_runs=max(1, int(milestone.get("max_runs", 5))),
        run_index=max(1, int(phase3.get("run_index", 1))),
        reporting_atom_type_mass=bool(reporting.get("atom_type_mass", True)),
    )


def run_phase3(
    config: dict[str, Any],
    seed: int,
    settings: Phase3Settings,
) -> dict[str, Any]:
    rng = random.Random(seed)
    runtime = _resolve_hf_runtime(settings=settings)
    synthetic_cfg = config.get("data", {}).get("synthetic", {})

    on_samples = generate_phase1_on_samples(config=synthetic_cfg, rng=rng, num_samples=settings.on_num_samples)
    off_samples_train = generate_phase1_off_samples(rng=rng, num_samples=settings.off_num_samples)
    fixed_spec = FixedStressEvalSpec(
        size=settings.off_fixed_stress_size,
        seed=settings.off_fixed_stress_seed,
        sampling=settings.off_fixed_stress_sampling,
        definition=settings.off_fixed_stress_definition,
    )
    scope_consistency = _validate_scope_consistency(settings=settings)
    off_samples_fixed = build_fixed_stress_eval_samples(spec=fixed_spec)

    gatea_artifacts: dict[str, Any] = {"gateA_kv_meta": [], "hf_runtime_meta": {}}
    gatea_records: list[dict[str, Any]] = []
    gatea = _run_gate_a_hf(
        on_samples=on_samples,
        off_samples=off_samples_train,
        settings=_gatea_compat(settings=settings),
        runtime=runtime,
        sample_records=gatea_records,
        artifacts=gatea_artifacts,
    )

    sample_records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {
        "gateA_kv_meta": gatea_artifacts.get("gateA_kv_meta", []),
        "hf_runtime_meta": gatea_artifacts.get("hf_runtime_meta", {}),
        "phase3_candidate_atoms": [],
        "phase3_state_pool_meta": [],
        "phase3_off_stress_top": [],
        "phase3_fixed_stress_eval_spec": {
            "size": fixed_spec.size,
            "seed": fixed_spec.seed,
            "sampling": fixed_spec.sampling,
            "definition": fixed_spec.definition,
        },
        "phase3_lambda_off_trace": [],
        "phase3_topk_schedule_trace": [],
        "phase3_loss_curve": [],
        "phase3_constraint_warn_trace": [],
        "phase3_atom_type_mass": [],
    }

    episode_states = _prepare_episode_train_states(
        runtime=runtime,
        on_samples=on_samples,
        settings=settings,
        artifacts=artifacts,
    )
    train_result = _optimize_episode_states(
        runtime=runtime,
        episode_states=episode_states,
        off_samples_train=off_samples_train,
        off_samples_fixed=off_samples_fixed,
        settings=settings,
    )
    artifacts["phase3_lambda_off_trace"] = train_result["lambda_trace"]
    artifacts["phase3_topk_schedule_trace"] = train_result["topk_trace"]
    artifacts["phase3_loss_curve"] = train_result["loss_curve"]
    artifacts["phase3_constraint_warn_trace"] = train_result["constraint_warn_trace"]

    state_pool = _build_phase3_state_pool(
        runtime=runtime,
        episode_states=episode_states,
        settings=settings,
        artifacts=artifacts,
    )
    on_eval = _evaluate_on_samples(
        runtime=runtime,
        on_samples=on_samples,
        state_pool=state_pool,
        settings=settings,
        sample_records=sample_records,
    )
    off_fixed_eval = _evaluate_off_samples(
        runtime=runtime,
        off_samples=off_samples_fixed,
        state_pool=state_pool,
        settings=settings,
        sample_records=sample_records,
        artifacts=artifacts,
        metric_prefix="phase3_delta_off",
    )
    off_train_eval = _evaluate_off_samples(
        runtime=runtime,
        off_samples=off_samples_train,
        state_pool=state_pool,
        settings=settings,
        sample_records=[],
        artifacts=None,
        metric_prefix="phase3_delta_off_train",
    )

    hard_pool = sorted(off_train_eval["stress_deltas"], reverse=True)[: settings.off_hard_pool_size]
    hard_pool_loss_off = cvar_tail_mean(hard_pool, settings.cvar_tail_fraction)
    fixed_stress_loss_off = cvar_tail_mean(off_fixed_eval["stress_deltas"], settings.cvar_tail_fraction)
    loss_off = select_report_off_loss(
        train_source=settings.off_train_source,
        hard_pool_loss=hard_pool_loss_off,
        fixed_stress_loss=fixed_stress_loss_off,
    )
    loss_on = on_eval["on_kl_mean"]
    loss_str = _structural_loss(state_pool=state_pool, settings=settings)

    lambda_trace = artifacts["phase3_lambda_off_trace"]
    constraint_warn_summary = dict(train_result.get("constraint_warn_summary", {}))
    constraint_warn_limit = float(settings.off_delta_p99_max + settings.constraint_warn_margin)
    constraint_warn_summary["final_eval_metric_name"] = "off_delta_p99_stress"
    constraint_warn_summary["final_eval_metric_source"] = "fixed_stress_eval"
    constraint_warn_summary["final_eval_metric_value"] = float(off_fixed_eval["off_delta_p99_stress"])
    constraint_warn_summary["final_eval_limit"] = float(constraint_warn_limit)
    constraint_warn_summary["final_eval_violated"] = bool(off_fixed_eval["off_delta_p99_stress"] > constraint_warn_limit)
    constraint_warn_summary["final_eval_typical_p99"] = float(off_fixed_eval["off_delta_p99_typical"])

    gateb = _build_gateb_result(
        settings=settings,
        on_eval=on_eval,
        off_eval=off_fixed_eval,
    )
    milestone = evaluate_phase3_outcome(
        gatea_pass=bool(gatea["pass"]),
        gateb_pass=bool(gateb["pass"]),
        explore_enabled=settings.milestone_explore_enabled,
        run_index=settings.run_index,
        max_runs=settings.milestone_explore_max_runs,
        on_gain_drop=settings.phase2_baseline_on_gain - on_eval["on_gain_mix"],
        off_tail_improve_ratio=_off_tail_improve_ratio(
            baseline=settings.phase2_baseline_off_p99,
            current=off_fixed_eval["off_delta_p99_stress"],
        ),
        on_gain_drop_max=settings.on_gain_drop_max,
        off_tail_improve_min=settings.off_tail_improve_min,
    )

    run_metrics = [
        ("phase3_on_acc_base", on_eval["on_acc_base"]),
        ("phase3_on_acc_teacher", on_eval["on_acc_teacher"]),
        ("phase3_on_acc_student_mix", on_eval["on_acc_student_mix"]),
        ("phase3_on_gain_mix", on_eval["on_gain_mix"]),
        ("phase3_rel_kl_to_teacher_mix", on_eval["rel_kl_to_teacher_mix"]),
        ("phase3_delta_on_mean_mix", on_eval["delta_on_mean_mix"]),
        ("phase3_off_delta_p99_stress", off_fixed_eval["off_delta_p99_stress"]),
        ("phase3_off_delta_p99_typical", off_fixed_eval["off_delta_p99_typical"]),
        ("phase3_off_delta_mean_stress", off_fixed_eval["off_delta_mean_stress"]),
        ("phase3_off_delta_mean_typical", off_fixed_eval["off_delta_mean_typical"]),
        ("phase3_loss_on", loss_on),
        ("phase3_loss_off", loss_off),
        ("phase3_loss_off_hard_pool_eval", hard_pool_loss_off),
        ("phase3_loss_off_fixed_stress_eval", fixed_stress_loss_off),
        ("phase3_loss_str", loss_str),
        ("phase3_lambda_off_last", float(lambda_trace[-1]["lambda_off"]) if lambda_trace else settings.lambda_off),
        ("phase3_constraint_source_is_fixed", 1.0 if settings.off_constraint_source == "fixed_stress_eval" else 0.0),
        ("phase3_dual_metric_source_is_fixed", 1.0 if settings.dual_metric_source == "fixed_stress_eval" else 0.0),
        ("phase3_constraint_warn_trigger_count", float(constraint_warn_summary.get("trigger_count", 0.0))),
        (
            "phase3_constraint_warn_max_consecutive_eval_ticks",
            float(constraint_warn_summary.get("max_consecutive_eval_violations", 0.0)),
        ),
        ("phase3_constraint_warn_final_eval_violation", 1.0 if constraint_warn_summary["final_eval_violated"] else 0.0),
        ("phase3_gateA_pass", 1.0 if gatea["pass"] else 0.0),
        ("phase3_gateB_pass", 1.0 if gateb["pass"] else 0.0),
        ("phase3_milestone_pass", 1.0 if milestone["milestone_pass"] else 0.0),
    ]

    report = {
        "phase": 3,
        "plan_version": "phase3_v3",
        "backend": "hf",
        "backend_reason": "hf runtime active",
        "device": settings.device,
        "model_id": settings.model_id,
        "decoding": settings.decoding,
        "policy_mode": settings.policy_mode,
        "policy_mode_effective": settings.policy_mode_effective,
        "ci_forced_prod": settings.ci_forced_prod,
        "debug_allow_nonfixed_constraint_sources": settings.debug_allow_nonfixed_constraint_sources,
        "mixture_scope": settings.layer_scope_mixture,
        "injection_scope": settings.layer_scope_injection,
        "mix_mode": settings.mix_mode,
        "k_anchor_policy": settings.k_anchor_policy,
        "scope_consistency": scope_consistency,
        "off_loss": {
            "type": settings.off_loss_type,
            "cvar_tail_fraction": settings.cvar_tail_fraction,
            "train_source": settings.off_train_source,
            "constraint_source": settings.off_constraint_source,
            "constraint_source_policy": "fixed_stress_eval_only_in_prod",
            "train_source_policy_note": "hard_pool allowed for train_source in prod/debug",
            "debug_override_enabled": settings.debug_allow_nonfixed_constraint_sources,
            "hard_pool_size": settings.off_hard_pool_size,
        },
        "fixed_stress_eval": artifacts["phase3_fixed_stress_eval_spec"],
        "dual_lambda": {
            "adaptive": settings.lambda_off_adaptive,
            "metric_source": settings.dual_metric_source,
            "ema_beta": settings.dual_ema_beta,
            "update_every": settings.dual_update_every,
            "lambda_min": settings.dual_lambda_min,
            "lambda_max": settings.dual_lambda_max,
            "delta_clip": settings.dual_delta_clip,
            "trace_len": len(lambda_trace),
        },
        "warnings": {"constraint_warn_summary": constraint_warn_summary},
        "gates": {"A": gatea, "B": gateb},
        "milestone": milestone,
        "sample_counts": {
            "on": len(on_samples),
            "off_train": len(off_samples_train),
            "off_fixed_stress": len(off_samples_fixed),
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


def build_fixed_stress_eval_samples(spec: FixedStressEvalSpec) -> list[dict[str, Any]]:
    if spec.size <= 0:
        raise ValueError("fixed_stress_eval.size must be > 0")
    if spec.sampling != "once_per_experiment":
        raise ValueError("Only fixed_stress_eval.sampling=once_per_experiment is supported")
    if spec.definition != "phase2_stress_compatible":
        raise ValueError("Only phase2_stress_compatible fixed_stress definition is supported")
    rng = random.Random(spec.seed)
    samples: list[dict[str, Any]] = []
    for i in range(spec.size):
        prompt = rng.choice(_FIXED_STRESS_PROMPTS)
        samples.append(
            {
                "sample_id": f"fixed_off_{spec.seed}_{i}",
                "episode_id": f"fixed_episode_{spec.seed}_{i}",
                "mode": "off",
                "answer": "",
                "demo_text": "",
                "query_text": prompt,
                "teacher_text": prompt,
            }
        )
    return samples


def cvar_tail_mean(values: list[float], tail_fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    tail_count = max(1, int(math.ceil(len(ordered) * tail_fraction)))
    tail = ordered[-tail_count:]
    return float(sum(tail) / len(tail))


def select_constraint_p99(metric_source: str, fixed_p99: float, hard_pool_p99: float) -> float:
    if metric_source == "fixed_stress_eval":
        return float(fixed_p99)
    if metric_source == "hard_pool":
        return float(hard_pool_p99)
    raise ValueError(f"Unsupported metric source: {metric_source}")


def select_report_off_loss(train_source: str, hard_pool_loss: float, fixed_stress_loss: float) -> float:
    if train_source == "hard_pool":
        return float(hard_pool_loss)
    if train_source == "fixed_stress_eval":
        return float(fixed_stress_loss)
    raise ValueError(f"Unsupported off_train_source: {train_source}")


def should_eval_constraint_tick(step: int, eval_every: int) -> bool:
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    return int(step) % int(eval_every) == 0


def update_constraint_warn_state(
    metric_value: float,
    threshold: float,
    margin: float,
    patience_eval_ticks: int,
    consecutive_violations: int,
) -> dict[str, Any]:
    if patience_eval_ticks <= 0:
        raise ValueError("patience_eval_ticks must be > 0")
    limit = float(threshold) + float(margin)
    violated = float(metric_value) > limit
    consecutive_next = int(consecutive_violations) + 1 if violated else 0
    triggered = bool(violated and consecutive_next >= int(patience_eval_ticks))
    return {
        "limit": limit,
        "violated": violated,
        "consecutive_violations": consecutive_next,
        "triggered": triggered,
    }


def _off_prompt_factors_tensor(
    torch_mod: Any,
    off_samples: list[dict[str, Any]],
    device: Any,
) -> Any:
    if not off_samples:
        return torch_mod.ones(1, dtype=torch_mod.float32, device=device)
    factors: list[float] = []
    for sample in off_samples:
        key = "|".join(
            (
                str(sample.get("sample_id", "")),
                str(sample.get("query_text", "")),
                str(sample.get("teacher_text", "")),
            )
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        ratio = int(digest[:8], 16) / float(0xFFFFFFFF)
        factors.append(0.9 + 0.2 * ratio)  # [0.9, 1.1] deterministic prompt factor
    return torch_mod.tensor(factors, dtype=torch_mod.float32, device=device)


def _stress_by_prompt(
    torch_mod: Any,
    state_values: Any,
    prompt_factors: Any,
) -> Any:
    flat = state_values.reshape(-1).to(dtype=torch_mod.float32)
    base = flat.mean()
    spread = flat.std(unbiased=False)
    norm = prompt_factors.reshape(-1)
    norm = norm / norm.mean().clamp_min(1e-6)
    stress = base * norm + 0.1 * spread * norm
    return stress.clamp_min(0.0)


def _hard_pool_tensor(torch_mod: Any, values: Any, pool_size: int) -> Any:
    flat = values.reshape(-1)
    if int(flat.shape[0]) <= max(1, pool_size):
        return flat
    topk_values, _ = torch_mod.topk(flat, k=max(1, min(int(flat.shape[0]), int(pool_size))))
    return topk_values


def run_dual_lambda_update_trace(
    steps: int,
    lambda_init: float,
    eta: float,
    threshold: float,
    p99_raw_values: list[float],
    ema_beta: float,
    update_every: int,
    lambda_min: float,
    lambda_max: float,
    delta_clip: float,
    adaptive: bool,
) -> list[dict[str, float]]:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if update_every <= 0:
        raise ValueError("update_every must be > 0")
    if not p99_raw_values:
        raise ValueError("p99_raw_values must not be empty")
    if not (0.0 <= ema_beta < 1.0):
        raise ValueError("ema_beta must be in [0, 1)")
    if lambda_max < lambda_min:
        raise ValueError("lambda_max must be >= lambda_min")

    trace: list[dict[str, float]] = []
    lambda_off = float(min(lambda_max, max(lambda_min, lambda_init)))
    p99_ema = float(p99_raw_values[0])
    for step in range(steps):
        raw = float(p99_raw_values[min(step, len(p99_raw_values) - 1)])
        p99_ema = ema_beta * p99_ema + (1.0 - ema_beta) * raw
        delta = 0.0
        if adaptive and step % update_every == 0:
            raw_delta = eta * (p99_ema - threshold)
            delta = float(max(-delta_clip, min(delta_clip, raw_delta)))
            lambda_off = float(max(lambda_min, min(lambda_max, lambda_off + delta)))
        trace.append(
            {
                "step": float(step),
                "p99_raw": raw,
                "p99_ema": p99_ema,
                "lambda_off": lambda_off,
                "delta": delta,
            }
        )
    return trace


def apply_delimiter_mass_constraints(
    weights: list[float],
    atom_types: list[str],
    mass_cap: float,
    mass_penalty: float,
) -> list[float]:
    if len(weights) != len(atom_types):
        raise ValueError("weights and atom_types length mismatch")
    if not weights:
        return []

    total = sum(max(0.0, value) for value in weights)
    if total <= 0.0:
        return [1.0 / len(weights) for _ in weights]
    base = [max(0.0, value) / total for value in weights]

    penalized = []
    for value, atom_type in zip(base, atom_types):
        if atom_type == "delimiter_only":
            penalized.append(value * (1.0 - mass_penalty))
        else:
            penalized.append(value)
    pen_total = sum(penalized)
    if pen_total <= 0.0:
        penalized = [1.0 / len(weights) for _ in weights]
    else:
        penalized = [value / pen_total for value in penalized]

    delimiter_ids = [idx for idx, atom_type in enumerate(atom_types) if atom_type == "delimiter_only"]
    if not delimiter_ids:
        return penalized

    delimiter_mass = sum(penalized[idx] for idx in delimiter_ids)
    if delimiter_mass <= mass_cap:
        return penalized

    non_ids = [idx for idx in range(len(atom_types)) if idx not in delimiter_ids]
    if not non_ids:
        return penalized
    target_delimiter = mass_cap
    target_non = 1.0 - target_delimiter

    scaled = list(penalized)
    if delimiter_mass > 0.0:
        scale_delim = target_delimiter / delimiter_mass
        for idx in delimiter_ids:
            scaled[idx] = penalized[idx] * scale_delim
    non_mass = sum(penalized[idx] for idx in non_ids)
    if non_mass <= 0.0:
        fill = target_non / len(non_ids)
        for idx in non_ids:
            scaled[idx] = fill
    else:
        scale_non = target_non / non_mass
        for idx in non_ids:
            scaled[idx] = penalized[idx] * scale_non
    final_total = sum(scaled)
    return [value / final_total for value in scaled]


def evaluate_phase3_outcome(
    gatea_pass: bool,
    gateb_pass: bool,
    explore_enabled: bool,
    run_index: int,
    max_runs: int,
    on_gain_drop: float,
    off_tail_improve_ratio: float,
    on_gain_drop_max: float,
    off_tail_improve_min: float,
) -> dict[str, Any]:
    milestone_applicable = bool(explore_enabled and run_index <= max_runs)
    milestone_pass = bool(
        milestone_applicable
        and on_gain_drop <= on_gain_drop_max
        and off_tail_improve_ratio >= off_tail_improve_min
    )
    return {
        "gate_pass": bool(gatea_pass and gateb_pass),
        "milestone_applicable": milestone_applicable,
        "milestone_pass": milestone_pass,
        "on_gain_drop": float(on_gain_drop),
        "on_gain_drop_max": float(on_gain_drop_max),
        "off_tail_improve_ratio": float(off_tail_improve_ratio),
        "off_tail_improve_min": float(off_tail_improve_min),
    }


def _resolve_hf_runtime(settings: Phase3Settings) -> HFModelRuntime:
    if settings.runtime_backend != "hf":
        raise HFBackendUnavailable(f"phase3 requires runtime backend 'hf', got: {settings.runtime_backend}")
    if settings.allow_mock_fallback:
        raise HFBackendUnavailable("phase3 disallows mock fallback; set allow_mock_fallback=false")
    return load_hf_model(model_id=settings.model_id, device=settings.device, torch_dtype=settings.torch_dtype)


def _validate_scope_consistency(settings: Phase3Settings) -> dict[str, Any]:
    allowed = {"top25", "all"}
    if settings.layer_scope_mixture not in allowed:
        raise ValueError(f"Unsupported phase3.layer_scope.mixture: {settings.layer_scope_mixture}")
    if settings.layer_scope_injection not in allowed:
        raise ValueError(f"Unsupported phase3.layer_scope.injection: {settings.layer_scope_injection}")
    if settings.mix_mode not in {"v_only", "kv_raw"}:
        raise ValueError(f"Unsupported phase3.mix.mode: {settings.mix_mode}")
    return {
        "checked": True,
        "mixture": settings.layer_scope_mixture,
        "injection": settings.layer_scope_injection,
        "mix_mode": settings.mix_mode,
        "consistent": True,
    }


def _prepare_episode_train_states(
    runtime: HFModelRuntime,
    on_samples: list[dict[str, Any]],
    settings: Phase3Settings,
    artifacts: dict[str, Any],
) -> list[_EpisodeTrainState]:
    states: list[_EpisodeTrainState] = []
    for sample in on_samples:
        atoms = _build_candidate_atoms(sample=sample, atom_types=settings.atom_types)
        atom_types = [atom["atom_type"] for atom in atoms]
        warm_scores = _warm_start_scores(
            atom_types=atom_types,
            anti_steer_strength=settings.anti_steer_strength,
        )
        warm_anchor_ids = _warm_anchor_ids_from_scores(
            scores=warm_scores,
            slot_count=settings.span_budget,
        )
        logits = _init_slot_logits(
            runtime=runtime,
            warm_scores=warm_scores,
            slot_count=settings.span_budget,
        )
        states.append(
            _EpisodeTrainState(
                sample=sample,
                atoms=atoms,
                atom_types=atom_types,
                warm_scores=warm_scores,
                warm_anchor_ids=warm_anchor_ids,
                logits=logits,
            )
        )
        artifacts["phase3_candidate_atoms"].append(
            {
                "sample_id": sample["sample_id"],
                "episode_id": sample["episode_id"],
                "atom_count": len(atoms),
                "atoms": atoms,
                "warm_scores": warm_scores,
                "warm_anchor_ids": warm_anchor_ids,
            }
        )
    if not states:
        raise ValueError("phase3 requires at least one ON sample")
    return states


def _optimize_episode_states(
    runtime: HFModelRuntime,
    episode_states: list[_EpisodeTrainState],
    settings: Phase3Settings,
    off_samples_train: list[dict[str, Any]] | None = None,
    off_samples_fixed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    torch_mod = runtime.torch
    params = [state.logits for state in episode_states]
    optimizer = torch_mod.optim.Adam(params, lr=settings.train_lr)
    device = episode_states[0].logits.device

    lambda_off_current = float(settings.lambda_off)
    p99_ema: float | None = None
    loss_curve: list[dict[str, float]] = []
    lambda_trace: list[dict[str, float]] = []
    topk_trace: list[dict[str, float]] = []
    constraint_warn_trace: list[dict[str, Any]] = []
    constraint_warn_eval_ticks = 0
    constraint_warn_consecutive = 0
    constraint_warn_max_consecutive = 0
    constraint_warn_trigger_count = 0
    train_inputs = off_samples_train or [{"sample_id": "off_train_default_0", "query_text": "default off train"}]
    fixed_inputs = off_samples_fixed or train_inputs
    train_factors = _off_prompt_factors_tensor(
        torch_mod=torch_mod,
        off_samples=train_inputs,
        device=device,
    )
    fixed_factors = _off_prompt_factors_tensor(
        torch_mod=torch_mod,
        off_samples=fixed_inputs,
        device=device,
    )

    for step in range(settings.train_steps):
        progress = 0.0 if settings.train_steps <= 1 else step / float(settings.train_steps - 1)
        topk = _topk_for_progress(
            progress=progress,
            schedule=settings.topk_schedule,
            milestones=settings.topk_milestones,
        )
        topk_trace.append(
            {
                "step": float(step),
                "progress": progress,
                "topk": float(topk),
            }
        )

        on_proxy_values: list[Any] = []
        off_proxy_values: list[Any] = []
        entropy_values: list[Any] = []
        overlap_values: list[Any] = []
        delimiter_overage_values: list[Any] = []

        for state in episode_states:
            slot_weights = _slot_weights_from_logits_tensor(
                torch_mod=torch_mod,
                logits=state.logits,
                topk=topk,
                atom_types=state.atom_types,
            )
            on_vec = _atom_type_vector(
                torch_mod=torch_mod,
                atom_types=state.atom_types,
                mapping={
                    "entry": 1.0,
                    "key_only": 0.85,
                    "value_only": 0.6,
                    "delimiter_only": 0.1,
                },
                device=state.logits.device,
            )
            off_vec = _atom_type_vector(
                torch_mod=torch_mod,
                atom_types=state.atom_types,
                mapping={
                    "entry": 0.25,
                    "key_only": 0.2,
                    "value_only": 0.55,
                    "delimiter_only": 1.0,
                },
                device=state.logits.device,
            )
            slot_on_scores = (slot_weights * on_vec).sum(dim=-1)
            slot_off_scores = (slot_weights * off_vec).sum(dim=-1)
            delimiter_mask = _delimiter_mask_tensor(
                torch_mod=torch_mod,
                atom_types=state.atom_types,
                device=state.logits.device,
            )
            delimiter_mass = (slot_weights * delimiter_mask).sum(dim=-1)
            delimiter_overage = torch_mod.relu(delimiter_mass - settings.delimiter_mass_cap).mean()
            entropy = _slot_entropy_mean(torch_mod=torch_mod, slot_weights=slot_weights)
            overlap = _slot_overlap_penalty(torch_mod=torch_mod, slot_weights=slot_weights)

            on_proxy_values.append(slot_on_scores.mean())
            off_proxy_values.append(slot_off_scores.max())  # stress-like proxy
            entropy_values.append(entropy)
            overlap_values.append(overlap)
            delimiter_overage_values.append(delimiter_overage)

        on_proxy = torch_mod.stack(on_proxy_values).mean()
        off_proxy_tensor = torch_mod.stack(off_proxy_values)
        train_stress = _stress_by_prompt(
            torch_mod=torch_mod,
            state_values=off_proxy_tensor,
            prompt_factors=train_factors,
        )
        fixed_stress = _stress_by_prompt(
            torch_mod=torch_mod,
            state_values=off_proxy_tensor,
            prompt_factors=fixed_factors,
        )
        hard_pool_tensor = _hard_pool_tensor(
            torch_mod=torch_mod,
            values=train_stress,
            pool_size=settings.off_hard_pool_size,
        )

        loss_on = -on_proxy
        if settings.off_train_source == "hard_pool":
            off_for_train = hard_pool_tensor
        elif settings.off_train_source == "fixed_stress_eval":
            off_for_train = fixed_stress
        else:
            raise ValueError(f"Unsupported off_train_source: {settings.off_train_source}")
        loss_off = _cvar_tail_mean_torch(torch_mod=torch_mod, values=off_for_train, tail_fraction=settings.cvar_tail_fraction)
        loss_str = (
            torch_mod.stack(entropy_values).mean()
            + torch_mod.stack(overlap_values).mean()
            + settings.delimiter_mass_penalty * torch_mod.stack(delimiter_overage_values).mean()
        )
        total_loss = settings.lambda_on * loss_on + lambda_off_current * loss_off + settings.lambda_str * loss_str

        optimizer.zero_grad()
        total_loss.backward()
        torch_mod.nn.utils.clip_grad_norm_(params, settings.train_grad_clip)
        optimizer.step()

        hard_pool_p99 = percentile([float(value) for value in hard_pool_tensor.detach().cpu().tolist()], 0.99)
        fixed_p99 = percentile([float(value) for value in fixed_stress.detach().cpu().tolist()], 0.99)
        constraint_p99 = select_constraint_p99(
            metric_source=settings.off_constraint_source,
            fixed_p99=fixed_p99,
            hard_pool_p99=hard_pool_p99,
        )
        p99_raw = select_constraint_p99(
            metric_source=settings.dual_metric_source,
            fixed_p99=fixed_p99,
            hard_pool_p99=hard_pool_p99,
        )
        constraint_warn_record: dict[str, Any] | None = None
        if settings.constraint_warn_enabled and should_eval_constraint_tick(step=step, eval_every=settings.constraint_eval_every):
            constraint_warn_eval_ticks += 1
            warn_update = update_constraint_warn_state(
                metric_value=float(fixed_p99),
                threshold=float(settings.off_delta_p99_max),
                margin=float(settings.constraint_warn_margin),
                patience_eval_ticks=int(settings.constraint_warn_patience_eval_ticks),
                consecutive_violations=int(constraint_warn_consecutive),
            )
            constraint_warn_consecutive = int(warn_update["consecutive_violations"])
            constraint_warn_max_consecutive = max(constraint_warn_max_consecutive, constraint_warn_consecutive)
            if bool(warn_update["triggered"]):
                constraint_warn_trigger_count += 1
            constraint_warn_record = {
                "step": float(step),
                "eval_tick": float(constraint_warn_eval_ticks - 1),
                "metric_name": "off_delta_p99_stress_proxy",
                "metric_source": "fixed_stress_eval",
                "metric_value": float(fixed_p99),
                "threshold": float(settings.off_delta_p99_max),
                "margin": float(settings.constraint_warn_margin),
                "limit": float(warn_update["limit"]),
                "violated": bool(warn_update["violated"]),
                "consecutive_violations": float(constraint_warn_consecutive),
                "triggered": bool(warn_update["triggered"]),
            }
            constraint_warn_trace.append(constraint_warn_record)

        if p99_ema is None:
            p99_ema = p99_raw
        else:
            p99_ema = settings.dual_ema_beta * p99_ema + (1.0 - settings.dual_ema_beta) * p99_raw
        delta = 0.0
        if settings.lambda_off_adaptive and (step % settings.dual_update_every == 0):
            raw_delta = settings.dual_eta * (p99_ema - settings.off_delta_p99_max)
            delta = max(-settings.dual_delta_clip, min(settings.dual_delta_clip, raw_delta))
            lambda_off_current = max(
                settings.dual_lambda_min,
                min(settings.dual_lambda_max, lambda_off_current + delta),
            )
        lambda_trace.append(
            {
                "step": float(step),
                "p99_raw": float(p99_raw),
                "p99_ema": float(p99_ema),
                "lambda_off": float(lambda_off_current),
                "delta": float(delta),
            }
        )
        loss_curve.append(
            {
                "step": float(step),
                "progress": progress,
                "topk": float(topk),
                "loss_total": float(total_loss.detach().cpu().item()),
                "loss_on": float(loss_on.detach().cpu().item()),
                "loss_off": float(loss_off.detach().cpu().item()),
                "loss_str": float(loss_str.detach().cpu().item()),
                "proxy_off_p99": float(p99_raw),
                "off_p99_constraint": float(constraint_p99),
                "off_p99_fixed": float(fixed_p99),
                "off_p99_hard_pool": float(hard_pool_p99),
                "off_train_source": settings.off_train_source,
                "off_constraint_source": settings.off_constraint_source,
                "dual_metric_source": settings.dual_metric_source,
                "constraint_warn_eval_tick": float(1.0 if constraint_warn_record is not None else 0.0),
                "constraint_warn_triggered": float(1.0 if constraint_warn_record and constraint_warn_record["triggered"] else 0.0),
                "constraint_warn_consecutive": float(constraint_warn_consecutive),
                "lambda_off": float(lambda_off_current),
            }
        )

    last_warn_metric = float(constraint_warn_trace[-1]["metric_value"]) if constraint_warn_trace else 0.0
    return {
        "lambda_trace": lambda_trace,
        "topk_trace": topk_trace,
        "loss_curve": loss_curve,
        "constraint_warn_trace": constraint_warn_trace,
        "constraint_warn_summary": {
            "enabled": bool(settings.constraint_warn_enabled),
            "metric_name": "off_delta_p99_stress_proxy",
            "metric_source": "fixed_stress_eval",
            "eval_every": float(settings.constraint_eval_every),
            "patience_eval_ticks": float(settings.constraint_warn_patience_eval_ticks),
            "margin": float(settings.constraint_warn_margin),
            "eval_ticks": float(constraint_warn_eval_ticks),
            "trigger_count": float(constraint_warn_trigger_count),
            "max_consecutive_eval_violations": float(constraint_warn_max_consecutive),
            "last_metric_value": last_warn_metric,
        },
    }


def _build_phase3_state_pool(
    runtime: HFModelRuntime,
    episode_states: list[_EpisodeTrainState],
    settings: Phase3Settings,
    artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    state_pool: list[dict[str, Any]] = []
    type_mass_agg: dict[str, float] = {}

    topk_final = _topk_for_progress(
        progress=1.0,
        schedule=settings.topk_schedule,
        milestones=settings.topk_milestones,
    )
    for idx, state in enumerate(episode_states):
        slot_weights = _final_slot_weights_from_logits(
            runtime=runtime,
            logits=state.logits,
            topk=topk_final,
            atom_types=state.atom_types,
            settings=settings,
        )
        anchor_ids = [
            select_k_anchor_index(
                weights=weights,
                warm_start_index=warm_anchor,
                policy=settings.k_anchor_policy,
            )
            for weights, warm_anchor in zip(slot_weights, state.warm_anchor_ids)
        ]
        k_anchor_fixed_ok = all(
            int(anchor_id) == int(warm_anchor_id)
            for anchor_id, warm_anchor_id in zip(anchor_ids, state.warm_anchor_ids)
        )

        for atom_type in ("entry", "key_only", "value_only", "delimiter_only"):
            avg = _average_atom_mass(
                slot_weights=slot_weights,
                atom_types=state.atom_types,
                target_type=atom_type,
            )
            type_mass_agg[atom_type] = type_mass_agg.get(atom_type, 0.0) + avg

        state_text = _compose_state_text_from_weights(
            atoms=state.atoms,
            slot_weights=slot_weights,
            fallback_anchor_ids=anchor_ids,
        )
        compact_text = _truncate_to_prefix_text(runtime=runtime, text=state_text, prefix_len=settings.prefix_len)
        capture = _capture_compact_prefix(runtime=runtime, compact_text=compact_text)
        state_pool.append(
            {
                "state_id": f"phase3_on_{idx}",
                "past_key_values": capture["past_key_values"],
                "past_len": capture["past_len"],
                "episode_id": state.sample["episode_id"],
                "atom_count": len(state.atoms),
                "anchor_ids": anchor_ids,
                "slot_weights": slot_weights,
                "atom_types": state.atom_types,
            }
        )
        artifacts["phase3_state_pool_meta"].append(
            {
                "state_id": f"phase3_on_{idx}",
                "episode_id": state.sample["episode_id"],
                "k_anchor_policy": settings.k_anchor_policy,
                "layer_scope_mixture": settings.layer_scope_mixture,
                "layer_scope_injection": settings.layer_scope_injection,
                "mix_mode": settings.mix_mode,
                "prefix_len": settings.prefix_len,
                "effective_prefix_len": capture["past_len"],
                "anchor_ids": anchor_ids,
                "warm_anchor_ids": state.warm_anchor_ids,
                "k_anchor_fixed_ok": k_anchor_fixed_ok,
                "topk_final": topk_final,
            }
        )

    if settings.reporting_atom_type_mass and state_pool:
        denom = float(len(state_pool))
        for atom_type, total in sorted(type_mass_agg.items()):
            artifacts["phase3_atom_type_mass"].append(
                {
                    "atom_type": atom_type,
                    "mean_mass": total / denom,
                }
            )
    if not state_pool:
        raise ValueError("phase3 requires at least one ON sample to build state pool")
    return state_pool


def _evaluate_on_samples(
    runtime: HFModelRuntime,
    on_samples: list[dict[str, Any]],
    state_pool: list[dict[str, Any]],
    settings: Phase3Settings,
    sample_records: list[dict[str, Any]],
) -> dict[str, float]:
    base_hits: list[float] = []
    teacher_hits: list[float] = []
    mix_hits: list[float] = []
    on_deltas: list[float] = []
    rel_kl_to_teacher_values: list[float] = []
    answer_logprob_gains: list[float] = []
    on_kl_values: list[float] = []

    for step, sample in enumerate(on_samples):
        query_text = str(sample["query_text"])
        teacher_text = _phase1_teacher_full_text(sample)
        state = state_pool[step % len(state_pool)]

        base = forward_next_token_logits(runtime=runtime, text=query_text, use_cache=True)
        teacher = forward_next_token_logits(runtime=runtime, text=teacher_text, use_cache=True)
        student = forward_next_token_logits_with_past(
            runtime=runtime,
            query_text=query_text,
            past_key_values=state["past_key_values"],
            past_len=state["past_len"],
        )

        answer_candidates = answer_first_token_candidate_ids(runtime=runtime, answer_text=str(sample["answer"]))
        base_hit = _answer_hit_from_candidates(logits=base["logits"], answer_candidates=answer_candidates)
        teacher_hit = _answer_hit_from_candidates(logits=teacher["logits"], answer_candidates=answer_candidates)
        mix_hit = _answer_hit_from_candidates(logits=student["logits"], answer_candidates=answer_candidates)

        delta_on = kl_from_logits(student["logits"], base["logits"])
        rel_kl_teacher = relative_kl(
            student_teacher_kl=kl_from_logits(student["logits"], teacher["logits"]),
            base_teacher_kl=kl_from_logits(base["logits"], teacher["logits"]),
        )
        on_kl = kl_from_logits(student["logits"], teacher["logits"])
        answer_gain = _answer_logprob_gain(
            runtime=runtime,
            answer_candidates=answer_candidates,
            selected_logits=student["logits"],
            base_logits=base["logits"],
        )

        base_hits.append(base_hit)
        teacher_hits.append(teacher_hit)
        mix_hits.append(mix_hit)
        on_deltas.append(delta_on)
        rel_kl_to_teacher_values.append(rel_kl_teacher)
        answer_logprob_gains.append(answer_gain)
        on_kl_values.append(on_kl)
        sample_records.extend(
            [
                _sample_metric(
                    sample=sample,
                    metric="phase3_delta_on_mix",
                    value=delta_on,
                    model_id=settings.model_id,
                    prefix_len=settings.prefix_len,
                    step=step,
                ),
                _sample_metric(
                    sample=sample,
                    metric="phase3_answer_logprob_gain",
                    value=answer_gain,
                    model_id=settings.model_id,
                    prefix_len=settings.prefix_len,
                    step=step,
                ),
            ]
        )

    return {
        "on_acc_base": sum(base_hits) / len(base_hits),
        "on_acc_teacher": sum(teacher_hits) / len(teacher_hits),
        "on_acc_student_mix": sum(mix_hits) / len(mix_hits),
        "on_gain_mix": (sum(mix_hits) / len(mix_hits)) - (sum(base_hits) / len(base_hits)),
        "delta_on_mean_mix": sum(on_deltas) / len(on_deltas),
        "rel_kl_to_teacher_mix": sum(rel_kl_to_teacher_values) / len(rel_kl_to_teacher_values),
        "answer_logprob_gain_mean": sum(answer_logprob_gains) / len(answer_logprob_gains),
        "on_kl_mean": sum(on_kl_values) / len(on_kl_values),
    }


def _evaluate_off_samples(
    runtime: HFModelRuntime,
    off_samples: list[dict[str, Any]],
    state_pool: list[dict[str, Any]],
    settings: Phase3Settings,
    sample_records: list[dict[str, Any]],
    artifacts: dict[str, Any] | None,
    metric_prefix: str,
) -> dict[str, Any]:
    if not state_pool:
        raise ValueError("state_pool must not be empty")
    stress_deltas: list[float] = []
    typical_deltas: list[float] = []

    for local_idx, sample in enumerate(off_samples):
        step = len(sample_records) + local_idx
        base = forward_next_token_logits(runtime=runtime, text=sample["query_text"], use_cache=True)
        stress_delta = -1.0
        stress_state_id = ""
        state_deltas: list[float] = []
        for state in state_pool:
            student = forward_next_token_logits_with_past(
                runtime=runtime,
                query_text=sample["query_text"],
                past_key_values=state["past_key_values"],
                past_len=state["past_len"],
            )
            delta = kl_from_logits(student["logits"], base["logits"])
            state_deltas.append(delta)
            if delta > stress_delta:
                stress_delta = delta
                stress_state_id = str(state["state_id"])

        rr_idx = local_idx % len(state_pool)
        typical_delta = state_deltas[rr_idx]
        stress_deltas.append(stress_delta)
        typical_deltas.append(typical_delta)
        if artifacts is not None:
            artifacts["phase3_off_stress_top"].append(
                {
                    "sample_id": sample["sample_id"],
                    "episode_id": sample["episode_id"],
                    "stress_state_id": stress_state_id,
                    "stress_delta": stress_delta,
                    "typical_state_id": str(state_pool[rr_idx]["state_id"]),
                    "typical_delta": typical_delta,
                }
            )
        if sample_records is not None:
            sample_records.extend(
                [
                    _sample_metric(
                        sample=sample,
                        metric=f"{metric_prefix}_stress_mix",
                        value=stress_delta,
                        model_id=settings.model_id,
                        prefix_len=settings.prefix_len,
                        step=step,
                    ),
                    _sample_metric(
                        sample=sample,
                        metric=f"{metric_prefix}_typical_mix",
                        value=typical_delta,
                        model_id=settings.model_id,
                        prefix_len=settings.prefix_len,
                        step=step,
                    ),
                ]
            )

    return {
        "stress_deltas": stress_deltas,
        "typical_deltas": typical_deltas,
        "off_delta_p99_stress": percentile(stress_deltas, 0.99),
        "off_delta_p99_typical": percentile(typical_deltas, 0.99),
        "off_delta_mean_stress": sum(stress_deltas) / len(stress_deltas),
        "off_delta_mean_typical": sum(typical_deltas) / len(typical_deltas),
    }


def _build_gateb_result(
    settings: Phase3Settings,
    on_eval: dict[str, float],
    off_eval: dict[str, float],
) -> dict[str, Any]:
    on_eval_valid = on_eval["on_acc_teacher"] >= settings.teacher_min_on_acc
    pass_core = _gateb_pass(
        on_gain=on_eval["on_gain_mix"],
        on_gain_min=settings.on_gain_min,
        off_delta_p99=off_eval["off_delta_p99_stress"],
        off_delta_p99_max=settings.off_delta_p99_max,
        delta_on_mean=on_eval["delta_on_mean_mix"],
        delta_on_min=settings.delta_on_min,
        rel_kl_to_teacher=on_eval["rel_kl_to_teacher_mix"],
        rel_kl_to_teacher_max=settings.rel_kl_to_teacher_max,
    )
    gate_pass = bool(on_eval_valid and pass_core)
    return {
        "required": True,
        "pass": gate_pass,
        "on_eval_valid": on_eval_valid,
        "teacher_min_on_acc": settings.teacher_min_on_acc,
        "on_acc_base": on_eval["on_acc_base"],
        "on_acc_teacher": on_eval["on_acc_teacher"],
        "on_acc_student_mix": on_eval["on_acc_student_mix"],
        "on_gain_mix": on_eval["on_gain_mix"],
        "on_gain_min": settings.on_gain_min,
        "delta_on_mean_mix": on_eval["delta_on_mean_mix"],
        "delta_on_min": settings.delta_on_min,
        "rel_kl_to_teacher_mix": on_eval["rel_kl_to_teacher_mix"],
        "rel_kl_to_teacher_max": settings.rel_kl_to_teacher_max,
        "off_delta_p99_stress": off_eval["off_delta_p99_stress"],
        "off_delta_p99_typical": off_eval["off_delta_p99_typical"],
        "off_delta_mean_stress": off_eval["off_delta_mean_stress"],
        "off_delta_mean_typical": off_eval["off_delta_mean_typical"],
        "off_delta_p99_max": settings.off_delta_p99_max,
        "failure_reason": _gateb_failure_reason(
            on_eval_valid=on_eval_valid,
            on_gain=on_eval["on_gain_mix"],
            on_gain_min=settings.on_gain_min,
            off_delta_p99=off_eval["off_delta_p99_stress"],
            off_delta_p99_max=settings.off_delta_p99_max,
            delta_on_mean=on_eval["delta_on_mean_mix"],
            delta_on_min=settings.delta_on_min,
            rel_kl_to_teacher=on_eval["rel_kl_to_teacher_mix"],
            rel_kl_to_teacher_max=settings.rel_kl_to_teacher_max,
        ),
    }


def _gateb_pass(
    on_gain: float,
    on_gain_min: float,
    off_delta_p99: float,
    off_delta_p99_max: float,
    delta_on_mean: float,
    delta_on_min: float,
    rel_kl_to_teacher: float,
    rel_kl_to_teacher_max: float,
) -> bool:
    return (
        on_gain >= on_gain_min
        and off_delta_p99 <= off_delta_p99_max
        and delta_on_mean >= delta_on_min
        and rel_kl_to_teacher <= rel_kl_to_teacher_max
    )


def _gateb_failure_reason(
    on_eval_valid: bool,
    on_gain: float,
    on_gain_min: float,
    off_delta_p99: float,
    off_delta_p99_max: float,
    delta_on_mean: float,
    delta_on_min: float,
    rel_kl_to_teacher: float,
    rel_kl_to_teacher_max: float,
) -> str:
    if not on_eval_valid:
        return "on_eval_invalid"
    if on_gain < on_gain_min:
        return "on_gain_below_min"
    if off_delta_p99 > off_delta_p99_max:
        return "off_delta_stress_p99_above_max"
    if delta_on_mean < delta_on_min:
        return "delta_on_below_min"
    if rel_kl_to_teacher > rel_kl_to_teacher_max:
        return "rel_kl_to_teacher_above_max"
    return "ok"


def _build_candidate_atoms(sample: dict[str, Any], atom_types: tuple[str, ...]) -> list[dict[str, str]]:
    demo_text = str(sample.get("demo_text", ""))
    atoms: list[dict[str, str]] = []
    for line in demo_text.splitlines():
        if "->" not in line:
            continue
        left, right = line.split("->", 1)
        key = left.strip()
        value = right.strip()
        if "entry" in atom_types:
            atoms.append({"atom_type": "entry", "text": f"{key} -> {value}\n"})
        if "key_only" in atom_types:
            atoms.append({"atom_type": "key_only", "text": f"{key}\n"})
        if "value_only" in atom_types:
            atoms.append({"atom_type": "value_only", "text": f"{value}\n"})
    if "delimiter_only" in atom_types:
        atoms.append({"atom_type": "delimiter_only", "text": " -> "})
        atoms.append({"atom_type": "delimiter_only", "text": "\n"})
    if not atoms:
        atoms.append({"atom_type": "entry", "text": demo_text})
    return atoms


def _warm_anchor_ids_from_scores(scores: list[float], slot_count: int) -> list[int]:
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    if not ranked:
        return [0 for _ in range(slot_count)]
    return [int(ranked[idx % len(ranked)]) for idx in range(slot_count)]


def _init_slot_logits(
    runtime: HFModelRuntime,
    warm_scores: list[float],
    slot_count: int,
) -> Any:
    torch_mod = runtime.torch
    device = runtime.model.device
    base = torch_mod.tensor(warm_scores, dtype=torch_mod.float32, device=device)
    tiled = base.unsqueeze(0).repeat(slot_count, 1)
    slot_bias = torch_mod.linspace(0.0, -0.2, steps=slot_count, dtype=torch_mod.float32, device=device).unsqueeze(1)
    param = torch_mod.nn.Parameter(tiled + slot_bias)
    return param


def _slot_weights_from_logits_tensor(
    torch_mod: Any,
    logits: Any,
    topk: int,
    atom_types: list[str],
) -> Any:
    probs = torch_mod.softmax(logits, dim=-1)
    topk_eff = max(1, min(int(topk), int(probs.shape[-1])))
    if topk_eff < int(probs.shape[-1]):
        _, indices = torch_mod.topk(probs, k=topk_eff, dim=-1)
        mask = torch_mod.zeros_like(probs)
        mask.scatter_(dim=-1, index=indices, value=1.0)
        probs = probs * mask
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def _atom_type_vector(
    torch_mod: Any,
    atom_types: list[str],
    mapping: dict[str, float],
    device: Any,
) -> Any:
    values = [float(mapping.get(atom_type, 0.0)) for atom_type in atom_types]
    return torch_mod.tensor(values, dtype=torch_mod.float32, device=device).unsqueeze(0)


def _delimiter_mask_tensor(torch_mod: Any, atom_types: list[str], device: Any) -> Any:
    values = [1.0 if atom_type == "delimiter_only" else 0.0 for atom_type in atom_types]
    return torch_mod.tensor(values, dtype=torch_mod.float32, device=device).unsqueeze(0)


def _slot_entropy_mean(torch_mod: Any, slot_weights: Any) -> Any:
    probs = slot_weights.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return entropy.mean()


def _slot_overlap_penalty(torch_mod: Any, slot_weights: Any) -> Any:
    if int(slot_weights.shape[0]) <= 1:
        return slot_weights.new_tensor(0.0)
    normalized = slot_weights / slot_weights.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = normalized @ normalized.transpose(0, 1)
    slot_count = int(slot_weights.shape[0])
    upper = torch_mod.triu(torch_mod.ones((slot_count, slot_count), device=slot_weights.device), diagonal=1)
    values = sim * upper
    denom = upper.sum().clamp_min(1.0)
    return values.sum() / denom


def _cvar_tail_mean_torch(torch_mod: Any, values: Any, tail_fraction: float) -> Any:
    sorted_values, _ = torch_mod.sort(values.reshape(-1))
    tail_count = max(1, int(math.ceil(int(sorted_values.shape[0]) * tail_fraction)))
    tail = sorted_values[-tail_count:]
    return tail.mean()


def _final_slot_weights_from_logits(
    runtime: HFModelRuntime,
    logits: Any,
    topk: int,
    atom_types: list[str],
    settings: Phase3Settings,
) -> list[list[float]]:
    torch_mod = runtime.torch
    with torch_mod.no_grad():
        weights = _slot_weights_from_logits_tensor(
            torch_mod=torch_mod,
            logits=logits,
            topk=topk,
            atom_types=atom_types,
        ).detach().cpu().tolist()
    if settings.delimiter_apply_stage != "post_topk_renorm_per_slot":
        raise ValueError(f"Unsupported delimiter apply stage: {settings.delimiter_apply_stage}")
    return [
        apply_delimiter_mass_constraints(
            weights=[float(value) for value in row],
            atom_types=atom_types,
            mass_cap=settings.delimiter_mass_cap,
            mass_penalty=settings.delimiter_mass_penalty,
        )
        for row in weights
    ]


def _compose_state_text_from_weights(
    atoms: list[dict[str, str]],
    slot_weights: list[list[float]],
    fallback_anchor_ids: list[int],
) -> str:
    selected_ids: list[int] = []
    for row in slot_weights:
        ranked = sorted(range(len(row)), key=lambda idx: row[idx], reverse=True)
        for idx in ranked[:2]:
            if idx not in selected_ids:
                selected_ids.append(idx)
    if not selected_ids:
        selected_ids = [idx for idx in fallback_anchor_ids if 0 <= idx < len(atoms)]
    if not selected_ids:
        selected_ids = list(range(min(1, len(atoms))))
    return "".join(str(atoms[idx]["text"]) for idx in selected_ids if 0 <= idx < len(atoms))


def _warm_start_scores(atom_types: list[str], anti_steer_strength: float) -> list[float]:
    base_score = {
        "entry": 1.0,
        "key_only": 0.8,
        "value_only": 0.65,
        "delimiter_only": 0.35,
    }
    steer_proxy = {
        "entry": 0.3,
        "key_only": 0.2,
        "value_only": 0.4,
        "delimiter_only": 1.0,
    }
    scores: list[float] = []
    for atom_type in atom_types:
        base = base_score.get(atom_type, 0.5)
        proxy = steer_proxy.get(atom_type, 0.5)
        scores.append(base - anti_steer_strength * proxy)
    return scores


def _build_slot_weights(
    scores: list[float],
    atom_types: list[str],
    slot_count: int,
    topk: int,
    delimiter_mass_cap: float,
    delimiter_mass_penalty: float,
    delimiter_apply_stage: str,
) -> tuple[list[list[float]], list[int]]:
    if not scores:
        raise ValueError("scores must not be empty")
    slot_weights: list[list[float]] = []
    anchor_ids: list[int] = []
    used_counts = [0 for _ in scores]

    for _ in range(slot_count):
        adjusted = [score - 0.1 * used_counts[idx] for idx, score in enumerate(scores)]
        probs = _softmax(adjusted)
        probs = _topk_renorm(probs, topk=max(1, min(topk, len(probs))))
        if delimiter_apply_stage == "post_topk_renorm_per_slot":
            probs = apply_delimiter_mass_constraints(
                weights=probs,
                atom_types=atom_types,
                mass_cap=delimiter_mass_cap,
                mass_penalty=delimiter_mass_penalty,
            )
        else:
            raise ValueError(f"Unsupported delimiter apply stage: {delimiter_apply_stage}")
        anchor = int(max(range(len(probs)), key=lambda i: probs[i]))
        used_counts[anchor] += 1
        slot_weights.append(probs)
        anchor_ids.append(anchor)
    return slot_weights, anchor_ids


def select_k_anchor_index(weights: list[float], warm_start_index: int, policy: str) -> int:
    if not weights:
        return 0
    if policy == "warm_start_fixed":
        if 0 <= warm_start_index < len(weights):
            return int(warm_start_index)
        return int(max(range(len(weights)), key=lambda i: weights[i]))
    raise ValueError(f"Unsupported k_anchor policy: {policy}")


def _softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(scores) for _ in scores]
    return [value / total for value in exps]


def _topk_renorm(weights: list[float], topk: int) -> list[float]:
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if topk >= len(weights):
        total = sum(weights)
        if total <= 0:
            return [1.0 / len(weights) for _ in weights]
        return [value / total for value in weights]
    ranked = sorted(range(len(weights)), key=lambda idx: weights[idx], reverse=True)
    kept = set(ranked[:topk])
    sparse = [weights[idx] if idx in kept else 0.0 for idx in range(len(weights))]
    total = sum(sparse)
    if total <= 0:
        fill = 1.0 / len(kept)
        return [fill if idx in kept else 0.0 for idx in range(len(weights))]
    return [value / total for value in sparse]


def _truncate_to_prefix_text(runtime: HFModelRuntime, text: str, prefix_len: int) -> str:
    encoded = runtime.tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = [int(token) for token in encoded][:prefix_len]
    return runtime.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)


def _answer_hit_from_candidates(logits: Any, answer_candidates: list[int]) -> float:
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


def _structural_loss(state_pool: list[dict[str, Any]], settings: Phase3Settings) -> float:
    entropies: list[float] = []
    delimiter_overages: list[float] = []
    for state in state_pool:
        slot_weights = state.get("slot_weights", [])
        atom_types = state.get("atom_types", [])
        for weights in slot_weights:
            entropy = 0.0
            for value in weights:
                p = max(1e-12, float(value))
                entropy -= p * math.log(p)
            entropies.append(entropy)
            delimiter_mass = 0.0
            for idx, atom_type in enumerate(atom_types):
                if atom_type == "delimiter_only":
                    delimiter_mass += float(weights[idx])
            delimiter_overages.append(max(0.0, delimiter_mass - settings.delimiter_mass_cap))
    entropy_mean = sum(entropies) / len(entropies) if entropies else 0.0
    delimiter_penalty = sum(delimiter_overages) / len(delimiter_overages) if delimiter_overages else 0.0
    return float(entropy_mean + settings.delimiter_mass_penalty * delimiter_penalty)


def _average_atom_mass(slot_weights: list[list[float]], atom_types: list[str], target_type: str) -> float:
    if not slot_weights:
        return 0.0
    total = 0.0
    for weights in slot_weights:
        slot_mass = 0.0
        for idx, atom_type in enumerate(atom_types):
            if atom_type == target_type:
                slot_mass += float(weights[idx])
        total += slot_mass
    return total / float(len(slot_weights))


def _off_tail_improve_ratio(baseline: float, current: float) -> float:
    if baseline <= 0:
        return 0.0
    return max(0.0, (baseline - current) / baseline)


def _build_topk_schedule_trace(settings: Phase3Settings) -> list[dict[str, float]]:
    trace: list[dict[str, float]] = []
    for step in range(settings.train_steps):
        progress = 0.0 if settings.train_steps <= 1 else step / float(settings.train_steps - 1)
        topk = _topk_for_progress(progress, settings.topk_schedule, settings.topk_milestones)
        trace.append(
            {
                "step": float(step),
                "progress": progress,
                "topk": float(topk),
            }
        )
    return trace


def _topk_for_progress(progress: float, schedule: tuple[int, ...], milestones: tuple[float, ...]) -> int:
    selected = schedule[0]
    for topk, milestone in zip(schedule, milestones):
        if progress >= milestone:
            selected = topk
    return int(selected)


@dataclass(frozen=True)
class _GateACompat:
    model_id: str
    prefix_len: int
    repro_runs: int
    teacher_min_divergence: float
    alpha_roundtrip: float
    eps_null_floor: float
    eps_null_multiplier: float
    eps_nonzero_multiplier: float
    eps_teacher_cache_floor: float
    eps_teacher_cache_multiplier: float


def _gatea_compat(settings: Phase3Settings) -> _GateACompat:
    return _GateACompat(
        model_id=settings.model_id,
        prefix_len=settings.prefix_len,
        repro_runs=settings.repro_runs,
        teacher_min_divergence=settings.teacher_min_divergence,
        alpha_roundtrip=settings.alpha_roundtrip,
        eps_null_floor=settings.eps_null_floor,
        eps_null_multiplier=settings.eps_null_multiplier,
        eps_nonzero_multiplier=settings.eps_nonzero_multiplier,
        eps_teacher_cache_floor=settings.eps_teacher_cache_floor,
        eps_teacher_cache_multiplier=settings.eps_teacher_cache_multiplier,
    )


def _parse_int_schedule(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",") if part.strip()]
        values = [int(token) for token in tokens]
    elif isinstance(raw, (list, tuple)):
        values = [int(item) for item in raw]
    else:
        values = [int(raw)]
    if not values:
        raise ValueError("topk schedule must not be empty")
    if any(value <= 0 for value in values):
        raise ValueError("topk schedule values must be > 0")
    return tuple(values)


def _parse_float_schedule(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",") if part.strip()]
        values = [float(token) for token in tokens]
    elif isinstance(raw, (list, tuple)):
        values = [float(item) for item in raw]
    else:
        values = [float(raw)]
    if not values:
        raise ValueError("topk milestones must not be empty")
    if values[0] != 0.0:
        values[0] = 0.0
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("topk milestones must be in [0, 1]")
    return tuple(values)
