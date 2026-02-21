from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cedkv_mvp.cli import _write_phase3_artifacts


class Phase3LossCurveArtifactWriteTest(unittest.TestCase):
    def test_loss_curve_artifact_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            payload = [{"step": 0.0, "loss_total": 1.23}]
            _write_phase3_artifacts(
                artifacts_dir=artifacts_dir,
                artifacts={"phase3_loss_curve": payload},
            )
            path = artifacts_dir / "phase3" / "loss_curve.json"
            self.assertTrue(path.exists())
            content = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(content, payload)


if __name__ == "__main__":
    unittest.main()
