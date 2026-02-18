from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from .config import load_config, write_yaml
from .metrics import MetricsWriter
from .runtime import build_run_id, get_environment_metadata, get_git_metadata, now_iso
from .synthetic import generate_synthetic_episode


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    git_sha, git_dirty = get_git_metadata()
    run_id = build_run_id(git_sha=git_sha, explicit_run_id=args.run_id)
    start_time = now_iso()
    env_meta = get_environment_metadata()

    output_root = Path(config.get("runtime", {}).get("output_root", "outputs"))
    run_dir = output_root / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    snapshot = _build_snapshot(
        config=config,
        run_id=run_id,
        start_time=start_time,
        git_sha=git_sha,
        git_dirty=git_dirty,
        env_meta=env_meta,
    )
    write_yaml(run_dir / "config.yaml", snapshot)

    seed = int(config.get("seed", 0))
    rng = random.Random(seed)
    synthetic_cfg = config.get("data", {}).get("synthetic", {})
    episode = generate_synthetic_episode(config=synthetic_cfg, rng=rng)

    with (artifacts_dir / "episode_000.json").open("w", encoding="utf-8") as f:
        json.dump(episode, f, ensure_ascii=True, indent=2)

    prompt_len = len(episode["prompt"].split())
    model_id = str(config.get("model", {}).get("id", "unknown-model"))

    writer = MetricsWriter(run_dir / "metrics.jsonl")
    writer.write(
        {
            "run_id": run_id,
            "step": 0,
            "split": "on",
            "mode": "on",
            "metric": "synthetic_episode_generated",
            "value": 1.0,
            "seed": seed,
            "timestamp": now_iso(),
            "model_id": model_id,
            "prompt_len": prompt_len,
            "prefix_len": 0,
            "episode_id": episode["episode_id"],
        }
    )

    writer.write(
        {
            "run_id": run_id,
            "step": 1,
            "split": "off",
            "mode": "off",
            "metric": "dry_run_completed" if args.dry_run else "run_completed",
            "value": 1.0,
            "seed": seed,
            "timestamp": now_iso(),
            "model_id": model_id,
            "prompt_len": prompt_len,
            "prefix_len": 0,
            "episode_id": episode["episode_id"],
        }
    )

    print(f"run_id={run_id}")
    print(f"output_dir={run_dir}")
    print(f"episode_id={episode['episode_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

