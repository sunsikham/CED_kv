from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


class SmokeTest(unittest.TestCase):
    def test_import_config_and_single_episode_dry_run(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "configs" / "base.yaml"

        sys.path.insert(0, str(repo_root))
        try:
            import cedkv_mvp  # noqa: F401
        finally:
            sys.path.pop(0)

        run_id = "smoke_test_run"
        run_dir = repo_root / "outputs" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "cedkv_mvp",
                "--config",
                str(config_path),
                "--dry-run",
                "--run-id",
                run_id,
            ],
            cwd=repo_root,
            check=True,
        )

        self.assertTrue((run_dir / "config.yaml").exists())
        self.assertTrue((run_dir / "metrics.jsonl").exists())
        self.assertTrue((run_dir / "artifacts").exists())

        with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("run_id", cfg)
        self.assertIn("start_time", cfg)
        self.assertIn("git_sha", cfg)
        self.assertIn("git_dirty", cfg)
        self.assertIn("host", cfg)
        self.assertIn("python_version", cfg)
        self.assertIn("torch_version", cfg)
        self.assertIn("cuda_version", cfg)
        self.assertIn("device_name", cfg)
        self.assertEqual(cfg["metrics_schema_version"], 1)

        with (run_dir / "metrics.jsonl").open("r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertGreaterEqual(len(lines), 1)
        required_fields = {
            "record_type",
            "run_id",
            "step",
            "split",
            "mode",
            "metric",
            "value",
            "seed",
            "timestamp",
            "model_id",
            "prompt_len",
            "prefix_len",
            "episode_id",
        }
        for record in lines:
            self.assertTrue(required_fields.issubset(record))
            self.assertTrue(record["episode_id"])
        self.assertTrue((run_dir / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
