from __future__ import annotations

import random
import uuid
from typing import Any


def generate_synthetic_episode(config: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    num_pairs = int(config.get("num_pairs", 5))
    key_prefix = str(config.get("key_prefix", "key"))
    value_prefix = str(config.get("value_prefix", "V"))

    mapping = {
        f"{key_prefix}_{i}": f"{value_prefix}{i}"
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


def _is_single_symbol(value: str) -> bool:
    return bool(value) and value.strip() == value and (" " not in value)


def generate_phase0_on_samples(
    config: dict[str, Any],
    rng: random.Random,
    num_samples: int,
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for _ in range(num_samples):
        episode = generate_synthetic_episode(config=config, rng=rng)
        answer = str(episode["answer"])
        if not _is_single_symbol(answer):
            raise ValueError(f"Phase0 ON answer must be a single symbol, got: {answer!r}")

        samples.append(
            {
                "sample_id": uuid.uuid4().hex,
                "episode_id": str(episode["episode_id"]),
                "mode": "on",
                "prompt": str(episode["prompt"]),
                "answer": answer,
            }
        )
    return samples


def generate_phase0_off_samples(
    rng: random.Random,
    num_samples: int,
) -> list[dict[str, str]]:
    prompts = [
        "Rewrite this sentence in a formal tone.",
        "Summarize the paragraph in one sentence.",
        "What is a good title for this note?",
        "Classify this text as positive or negative.",
        "Translate this sentence to Korean.",
    ]

    samples: list[dict[str, str]] = []
    for _ in range(num_samples):
        prompt = rng.choice(prompts)
        samples.append(
            {
                "sample_id": uuid.uuid4().hex,
                "episode_id": uuid.uuid4().hex,
                "mode": "off",
                "prompt": prompt,
                "answer": "",
            }
        )
    return samples


def generate_phase1_on_samples(
    config: dict[str, Any],
    rng: random.Random,
    num_samples: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for _ in range(num_samples):
        episode = generate_synthetic_episode(config=config, rng=rng)
        answer = str(episode["answer"])
        if not _is_single_symbol(answer):
            raise ValueError(f"Phase1 ON answer must be a single symbol, got: {answer!r}")

        mapping = episode["mapping"]
        demo_lines = [f"{k} -> {v}" for k, v in mapping.items()]
        demo_text = "\n".join(demo_lines)
        question = f"What is the value for {episode['query_key']}?"
        query_text = f"Question: {question}\nAnswer:"
        full_teacher_text = f"Demo:\n{demo_text}\n\n{query_text}"
        samples.append(
            {
                "sample_id": uuid.uuid4().hex,
                "episode_id": str(episode["episode_id"]),
                "mode": "on",
                "answer": answer,
                "demo_text": demo_text,
                "query_text": query_text,
                "teacher_text": full_teacher_text,
            }
        )
    return samples


def generate_phase1_off_samples(
    rng: random.Random,
    num_samples: int,
) -> list[dict[str, Any]]:
    prompts = [
        "Rewrite this sentence in a formal tone.",
        "Summarize this paragraph in one sentence.",
        "Classify sentiment as positive or negative.",
        "Translate this sentence to Korean.",
        "List three concise title options.",
    ]
    samples: list[dict[str, Any]] = []
    for _ in range(num_samples):
        prompt = rng.choice(prompts)
        samples.append(
            {
                "sample_id": uuid.uuid4().hex,
                "episode_id": uuid.uuid4().hex,
                "mode": "off",
                "answer": "",
                "demo_text": "",
                "query_text": prompt,
                "teacher_text": prompt,
            }
        )
    return samples
