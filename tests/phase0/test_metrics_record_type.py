from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


class MetricsRecordTypeTest(unittest.TestCase):
    def test_metrics_contains_run_and_sample_record_types(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "configs" / "base.yaml"
        run_id = "phase0_record_type"
        run_dir = repo_root / "outputs" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "cedkv_mvp",
                "--config",
                str(config_path),
                "--phase",
                "0",
                "--stub-mode",
                "null",
                "--run-id",
                run_id,
            ],
            cwd=repo_root,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)

        metrics_path = run_dir / "metrics.jsonl"
        self.assertTrue(metrics_path.exists())
        with metrics_path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        record_types = {record["record_type"] for record in records}
        self.assertEqual(record_types, {"run", "sample"})


if __name__ == "__main__":
    unittest.main()

