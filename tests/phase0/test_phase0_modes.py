from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


def _run_phase0(repo_root: Path, run_id: str, stub_mode: str) -> tuple[int, dict]:
    config_path = repo_root / "configs" / "base.yaml"
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
            stub_mode,
            "--run-id",
            run_id,
        ],
        cwd=repo_root,
        check=False,
    )
    report_path = run_dir / "report.json"
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    return completed.returncode, report


class Phase0ModesTest(unittest.TestCase):
    def test_null_stub_passes_gate(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        return_code, report = _run_phase0(
            repo_root=repo_root,
            run_id="phase0_null_pass",
            stub_mode="null",
        )
        self.assertEqual(return_code, 0)
        self.assertTrue(report["gates"]["pass"])

    def test_selective_stub_improves_on_metric_and_passes(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        return_code, report = _run_phase0(
            repo_root=repo_root,
            run_id="phase0_selective_pass",
            stub_mode="selective",
        )
        self.assertEqual(return_code, 0)
        self.assertTrue(report["gates"]["pass"])
        self.assertGreater(
            report["on_metrics"]["on_exact_match_acc_student"],
            report["on_metrics"]["on_exact_match_acc_base"],
        )


if __name__ == "__main__":
    unittest.main()

