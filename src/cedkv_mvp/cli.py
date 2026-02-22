from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .config import load_config, write_yaml
from .eval_phase0 import resolve_phase0_settings, run_phase0
from .eval_phase1 import resolve_phase1_settings, run_phase1
from .eval_phase3 import resolve_phase3_settings, run_phase3
from .metrics import MetricsWriter
from .report import write_report
from .runtime import build_run_id, get_environment_metadata, get_git_metadata, now_iso


def _build_snapshot(
    config: dict[str, Any],
    run_id: str,
    start_time: str,
    git_sha: str,
    git_dirty: bool,
    env_meta: dict[str, str],
) -> dict[str, Any]:
    snapshot = copy.deepcopy(config)
    snapshot["run_id"] = run_id
    snapshot["start_time"] = start_time
    snapshot["git_sha"] = git_sha
    snapshot["git_dirty"] = git_dirty
    snapshot["host"] = env_meta["host"]
    snapshot["python_version"] = env_meta["python_version"]
    snapshot["torch_version"] = env_meta["torch_version"]
    snapshot["cuda_version"] = env_meta["cuda_version"]
    snapshot["device_name"] = env_meta["device_name"]
    snapshot["metrics_schema_version"] = int(
        config.get("metrics", {}).get("schema_version", 1)
    )
    return snapshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CED-KV MVP CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional fixed run id. If omitted, uses timestamp+shortsha",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run minimal synthetic pipeline without model inference",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        help="Execution phase. Supported values: 0, 1, 2, 3.",
    )
    parser.add_argument(
        "--stub-mode",
        choices=("null", "global", "selective"),
        default=None,
        help="Phase0 override for stub predictor mode.",
    )
    parser.add_argument(
        "--max-on-samples",
        type=int,
        default=None,
        help="Phase0 ON sample count override.",
    )
    parser.add_argument(
        "--max-off-samples",
        type=int,
        default=None,
        help="Phase0 OFF sample count override.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional explicit report output path.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional model id override for phase1.",
    )
    parser.add_argument(
        "--phase1-device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Phase1 device override.",
    )
    parser.add_argument(
        "--phase1-max-on",
        type=int,
        default=None,
        help="Phase1 ON sample count override.",
    )
    parser.add_argument(
        "--phase1-max-off",
        type=int,
        default=None,
        help="Phase1 OFF sample count override.",
    )
    parser.add_argument(
        "--phase1-prefix-len",
        type=int,
        default=None,
        help="Phase1 prefix length override.",
    )
    parser.add_argument(
        "--phase1-select-mode",
        type=str,
        choices=("attention_diversity_span", "delimiter_span", "random_span"),
        default=None,
        help="Phase1 span selection mode override.",
    )
    parser.add_argument(
        "--phase1-backend",
        type=str,
        choices=("mock", "hf"),
        default=None,
        help="Phase1 backend override.",
    )
    parser.add_argument(
        "--phase1-strict-hf",
        action="store_true",
        help="Disable mock fallback when phase1 backend is hf.",
    )
    parser.add_argument(
        "--phase1-dtype",
        type=str,
        choices=("auto", "float16", "bfloat16", "float32"),
        default=None,
        help="Phase1 HF torch dtype override.",
    )
    parser.add_argument(
        "--phase2-model-id",
        type=str,
        default=None,
        help="Optional model id override for phase2.",
    )
    parser.add_argument(
        "--phase2-device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Phase2 device override.",
    )
    parser.add_argument(
        "--phase2-max-on",
        type=int,
        default=None,
        help="Phase2 ON sample count override.",
    )
    parser.add_argument(
        "--phase2-max-off",
        type=int,
        default=None,
        help="Phase2 OFF sample count override.",
    )
    parser.add_argument(
        "--phase2-prefix-len",
        type=int,
        default=None,
        help="Phase2 prefix length override.",
    )
    parser.add_argument(
        "--phase2-select-mode",
        type=str,
        choices=("attention_diversity_span", "delimiter_span", "random_span", "causal_drop_span"),
        default=None,
        help="Phase2 selection mode override.",
    )
    parser.add_argument(
        "--phase2-strict-hf",
        action="store_true",
        help="Disable mock fallback for phase2.",
    )
    parser.add_argument(
        "--phase2-dtype",
        type=str,
        choices=("auto", "float16", "bfloat16", "float32"),
        default=None,
        help="Phase2 HF torch dtype override.",
    )
    parser.add_argument(
        "--phase3-model-id",
        type=str,
        default=None,
        help="Optional model id override for phase3.",
    )
    parser.add_argument(
        "--phase3-device",
        type=str,
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Phase3 device override.",
    )
    parser.add_argument(
        "--phase3-max-on",
        type=int,
        default=None,
        help="Phase3 ON sample count override.",
    )
    parser.add_argument(
        "--phase3-max-off",
        type=int,
        default=None,
        help="Phase3 OFF train sample count override.",
    )
    parser.add_argument(
        "--phase3-prefix-len",
        type=int,
        default=None,
        help="Phase3 prefix length override.",
    )
    parser.add_argument(
        "--phase3-strict-hf",
        action="store_true",
        help="Disable mock fallback for phase3.",
    )
    parser.add_argument(
        "--phase3-dtype",
        type=str,
        choices=("auto", "float16", "bfloat16", "float32"),
        default=None,
        help="Phase3 HF torch dtype override.",
    )
    parser.add_argument(
        "--phase3-injection-scope",
        type=str,
        choices=("top25", "all"),
        default=None,
        help="Phase3 injection scope override.",
    )
    parser.add_argument(
        "--phase3-mix-mode",
        type=str,
        choices=("v_only", "kv_raw"),
        default=None,
        help="Phase3 mix mode override.",
    )
    parser.add_argument(
        "--phase3-off-loss-type",
        type=str,
        choices=("cvar",),
        default=None,
        help="Phase3 off loss type override.",
    )
    parser.add_argument(
        "--phase3-cvar-tail-fraction",
        type=float,
        default=None,
        help="Phase3 CVaR tail fraction override.",
    )
    parser.add_argument(
        "--phase3-topk-schedule",
        type=str,
        default=None,
        help="Phase3 top-k schedule override, e.g. 32,16,8,4",
    )
    parser.add_argument(
        "--phase3-lambda-off-adaptive",
        action="store_true",
        help="Enable adaptive lambda_off for phase3.",
    )
    parser.add_argument(
        "--phase3-explore",
        action="store_true",
        help="Enable phase3 exploration milestone mode.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase not in (0, 1, 2, 3):
        raise ValueError(f"Unsupported --phase {args.phase}. Supported values: 0, 1, 2, 3.")
    config = load_config(args.config)

    git_sha, git_dirty = get_git_metadata()
    run_id = build_run_id(git_sha=git_sha, explicit_run_id=args.run_id)
    start_time = now_iso()
    env_meta = get_environment_metadata()

    output_root = Path(config.get("runtime", {}).get("output_root", "outputs"))
    run_dir = output_root / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    snapshot = _build_snapshot(
        config=config,
        run_id=run_id,
        start_time=start_time,
        git_sha=git_sha,
        git_dirty=git_dirty,
        env_meta=env_meta,
    )
    snapshot["phase"] = args.phase
    write_yaml(run_dir / "config.yaml", snapshot)

    if args.phase == 0:
        seed = int(config.get("phase0", {}).get("seed", config.get("seed", 0)))
        model_id = str(config.get("model", {}).get("id", "unknown-model"))
        settings = resolve_phase0_settings(
            config=config,
            stub_mode_override=args.stub_mode,
            max_on_samples=args.max_on_samples,
            max_off_samples=args.max_off_samples,
        )
        result = run_phase0(
            config=config,
            seed=seed,
            model_id=model_id,
            settings=settings,
        )
        report_name = "report.json"
        extra_prints = {
            "stub_mode": settings.stub_mode,
            "gate_pass": result["gate_pass"],
        }
    elif args.phase == 1:
        seed = int(config.get("phase1", {}).get("seed", config.get("seed", 0)))
        settings = resolve_phase1_settings(
            config=config,
            model_id_override=args.model_id,
            device_override=args.phase1_device,
            max_on_samples=args.phase1_max_on,
            max_off_samples=args.phase1_max_off,
            prefix_len_override=args.phase1_prefix_len,
            select_mode_override=args.phase1_select_mode,
            backend_override=args.phase1_backend,
            strict_hf_override=True if args.phase1_strict_hf else None,
            dtype_override=args.phase1_dtype,
        )
        try:
            result = run_phase1(
                config=config,
                seed=seed,
                settings=settings,
            )
        except Exception as exc:
            report_path = args.report_path or (run_dir / "report_phase1.json")
            write_report(
                path=report_path,
                report={
                    "run_id": run_id,
                    "phase": 1,
                    "plan_version": "phase15_v1",
                    "error": str(exc),
                    "gateA_pass": False,
                    "gateB_pass": False,
                },
            )
            print(f"run_id={run_id}")
            print(f"output_dir={run_dir}")
            print("gateA_pass=False")
            print("gateB_pass=False")
            print(f"report_path={report_path}")
            return 2
        report_name = "report_phase1.json"
        _write_phase1_artifacts(artifacts_dir=artifacts_dir, artifacts=result.get("artifacts", {}))
        extra_prints = {
            "gateA_pass": result["gateA_pass"],
            "gateB_pass": result["gateB_pass"],
            "select_mode": settings.select_mode,
        }
        model_id = settings.model_id
    elif args.phase == 2:
        from .eval_phase2 import resolve_phase2_settings, run_phase2

        seed = int(config.get("phase2", {}).get("seed", config.get("seed", 0)))
        settings = resolve_phase2_settings(
            config=config,
            model_id_override=args.phase2_model_id,
            device_override=args.phase2_device,
            max_on_samples=args.phase2_max_on,
            max_off_samples=args.phase2_max_off,
            prefix_len_override=args.phase2_prefix_len,
            select_mode_override=args.phase2_select_mode,
            strict_hf_override=True if args.phase2_strict_hf else None,
            dtype_override=args.phase2_dtype,
        )
        try:
            result = run_phase2(
                config=config,
                seed=seed,
                settings=settings,
            )
        except Exception as exc:
            report_path = args.report_path or (run_dir / "report_phase2.json")
            write_report(
                path=report_path,
                report={
                    "run_id": run_id,
                    "phase": 2,
                    "plan_version": "phase2_v2",
                    "error": str(exc),
                    "gateA_pass": False,
                    "gateB_pass": False,
                },
            )
            print(f"run_id={run_id}")
            print(f"output_dir={run_dir}")
            print("gateA_pass=False")
            print("gateB_pass=False")
            print(f"report_path={report_path}")
            return 2
        report_name = "report_phase2.json"
        _write_phase2_artifacts(artifacts_dir=artifacts_dir, artifacts=result.get("artifacts", {}))
        extra_prints = {
            "gateA_pass": result["gateA_pass"],
            "gateB_pass": result["gateB_pass"],
            "select_mode": settings.select_mode,
            "positioning_mode": settings.positioning_mode,
        }
        model_id = settings.model_id
    else:
        seed = int(config.get("phase3", {}).get("seed", config.get("seed", 0)))
        settings = resolve_phase3_settings(
            config=config,
            model_id_override=args.phase3_model_id,
            device_override=args.phase3_device,
            max_on_samples=args.phase3_max_on,
            max_off_samples=args.phase3_max_off,
            prefix_len_override=args.phase3_prefix_len,
            strict_hf_override=True if args.phase3_strict_hf else None,
            dtype_override=args.phase3_dtype,
            injection_scope_override=args.phase3_injection_scope,
            mix_mode_override=args.phase3_mix_mode,
            off_loss_type_override=args.phase3_off_loss_type,
            cvar_tail_fraction_override=args.phase3_cvar_tail_fraction,
            topk_schedule_override=args.phase3_topk_schedule,
            lambda_off_adaptive_override=True if args.phase3_lambda_off_adaptive else None,
            explore_override=True if args.phase3_explore else None,
        )
        try:
            result = run_phase3(
                config=config,
                seed=seed,
                settings=settings,
            )
        except Exception as exc:
            report_path = args.report_path or (run_dir / "report_phase3.json")
            write_report(
                path=report_path,
                report={
                    "run_id": run_id,
                    "phase": 3,
                    "plan_version": "phase3_v3",
                    "error": str(exc),
                    "gateA_pass": False,
                    "gateB_pass": False,
                },
            )
            print(f"run_id={run_id}")
            print(f"output_dir={run_dir}")
            print("gateA_pass=False")
            print("gateB_pass=False")
            print(f"report_path={report_path}")
            return 2
        report_name = "report_phase3.json"
        _write_phase3_artifacts(artifacts_dir=artifacts_dir, artifacts=result.get("artifacts", {}))
        extra_prints = {
            "gateA_pass": result["gateA_pass"],
            "gateB_pass": result["gateB_pass"],
            "mix_mode": settings.mix_mode,
            "injection_scope": settings.layer_scope_injection,
        }
        model_id = settings.model_id

    _write_metrics(
        metrics_path=metrics_path,
        run_id=run_id,
        seed=seed,
        model_id=model_id,
        result=result,
    )
    report = {"run_id": run_id, **result["report"]}
    report_path = args.report_path or (run_dir / report_name)
    write_report(path=report_path, report=report)

    print(f"run_id={run_id}")
    print(f"output_dir={run_dir}")
    for key, value in extra_prints.items():
        print(f"{key}={value}")
    print(f"report_path={report_path}")

    return 0 if result["gate_pass"] else 2


def _write_metrics(
    metrics_path: Path,
    run_id: str,
    seed: int,
    model_id: str,
    result: dict[str, Any],
) -> None:
    writer = MetricsWriter(metrics_path)
    for record in result["sample_records"]:
        writer.write(
            {
                "run_id": run_id,
                "seed": seed,
                **record,
            }
        )
    for step, (metric_name, metric_value) in enumerate(result["run_metrics"], start=1):
        writer.write(
            {
                "record_type": "run",
                "run_id": run_id,
                "step": step,
                "split": "all",
                "mode": "all",
                "metric": metric_name,
                "value": metric_value,
                "seed": seed,
                "timestamp": now_iso(),
                "model_id": model_id,
                "prompt_len": 0,
                "prefix_len": 0,
                "episode_id": "run",
            }
        )


def _write_phase1_artifacts(artifacts_dir: Path, artifacts: dict[str, Any]) -> None:
    phase1_dir = artifacts_dir / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "gateA_kv_meta": phase1_dir / "kv_capture_meta.json",
        "gateB_selection": phase1_dir / "selected_spans.json",
        "hf_runtime_meta": phase1_dir / "hf_runtime_meta.json",
    }
    for key, path in mapping.items():
        payload = artifacts.get(key, [])
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
            f.write("\n")


def _write_phase2_artifacts(artifacts_dir: Path, artifacts: dict[str, Any]) -> None:
    phase2_dir = artifacts_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "gateA_kv_meta": phase2_dir / "gateA_kv_meta.json",
        "phase2_selection": phase2_dir / "phase2_selection.json",
        "phase2_prefix_meta": phase2_dir / "phase2_prefix_meta.json",
        "phase2_off_stress_top": phase2_dir / "phase2_off_stress_top.json",
        "hf_runtime_meta": phase2_dir / "hf_runtime_meta.json",
    }
    for key, path in mapping.items():
        payload = artifacts.get(key, [])
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
            f.write("\n")


def _write_phase3_artifacts(artifacts_dir: Path, artifacts: dict[str, Any]) -> None:
    phase3_dir = artifacts_dir / "phase3"
    phase3_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "gateA_kv_meta": phase3_dir / "gateA_kv_meta.json",
        "phase3_candidate_atoms": phase3_dir / "candidate_atoms.json",
        "phase3_state_pool_meta": phase3_dir / "state_pool_meta.json",
        "phase3_off_stress_top": phase3_dir / "off_stress_top.json",
        "phase3_fixed_stress_eval_spec": phase3_dir / "fixed_stress_eval_spec.json",
        "phase3_lambda_off_trace": phase3_dir / "lambda_off_trace.json",
        "phase3_topk_schedule_trace": phase3_dir / "topk_schedule_trace.json",
        "phase3_loss_curve": phase3_dir / "loss_curve.json",
        "phase3_constraint_warn_trace": phase3_dir / "constraint_warn_trace.json",
        "phase3_atom_type_mass": phase3_dir / "atom_type_mass.json",
        "hf_runtime_meta": phase3_dir / "hf_runtime_meta.json",
    }
    for key, path in mapping.items():
        payload = artifacts.get(key, [])
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
            f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
