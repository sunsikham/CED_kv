from __future__ import annotations

import random
import uuid
from typing import Any


def generate_synthetic_episode(config: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    num_pairs = int(config.get("num_pairs", 5))
    key_prefix = str(config.get("key_prefix", "key"))
    value_prefix = str(config.get("value_prefix", "value"))

    mapping = {
        f"{key_prefix}_{i}": f"{value_prefix}_{i}"
        for i in range(num_pairs)
    }
    keys = list(mapping.keys())
    query_key = rng.choice(keys)
    answer = mapping[query_key]
    context = "\n".join(f"{k} -> {v}" for k, v in mapping.items())
    prompt = (
        "You are given key-value pairs.\n"
        f"{context}\n\n"
        f"Question: What is the value for {query_key}?\n"
        "Answer:"
    )

    return {
        "episode_id": uuid.uuid4().hex,
        "mapping": mapping,
        "query_key": query_key,
        "answer": answer,
        "prompt": prompt,
    }

