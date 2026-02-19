from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


class Phase0GateFailTest(unittest.TestCase):
    def test_global_stub_fails_gate_with_exit_2(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "configs" / "base.yaml"
        run_id = "phase0_global_fail"
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
                "global",
                "--run-id",
                run_id,
            ],
            cwd=repo_root,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)

        report_path = run_dir / "report.json"
        self.assertTrue(report_path.exists())
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        self.assertFalse(report["gates"]["pass"])


if __name__ == "__main__":
    unittest.main()

