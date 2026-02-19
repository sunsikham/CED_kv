from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_METRIC_FIELDS = (
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
)


class MetricsWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        missing = [field for field in REQUIRED_METRIC_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Metric record missing required fields: {missing}")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")
