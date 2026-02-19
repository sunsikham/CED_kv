from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Sample = dict[str, str]
Distribution = dict[str, float]
Predictor = Callable[[Sample], Distribution]


GLOBAL_STEER_TOKEN = "GLOBAL_STEER"
BASE_DEFAULT_TOKEN = "BASE_DEFAULT"


def _normalize(dist: Distribution) -> Distribution:
    total = sum(max(value, 0.0) for value in dist.values())
    if total <= 0:
        uniform = 1.0 / len(dist)
        return {token: uniform for token in dist}
    return {token: max(value, 0.0) / total for token, value in dist.items()}


def _base_distribution(sample: Sample, vocab: list[str]) -> Distribution:
    answer = sample.get("answer", "")
    dist = {token: 0.01 for token in vocab}
    if answer in dist:
        dist[answer] = 0.25
    if BASE_DEFAULT_TOKEN in dist:
        dist[BASE_DEFAULT_TOKEN] = 0.60
    return _normalize(dist)


def _teacher_distribution(sample: Sample, vocab: list[str]) -> Distribution:
    base = _base_distribution(sample, vocab)
    if sample.get("mode") != "on":
        return base

    answer = sample.get("answer", "")
    dist = {token: 0.01 for token in vocab}
    if answer in dist:
        dist[answer] = 0.90
    if BASE_DEFAULT_TOKEN in dist:
        dist[BASE_DEFAULT_TOKEN] = 0.02
    return _normalize(dist)


def _global_distribution(vocab: list[str]) -> Distribution:
    dist = {token: 0.01 for token in vocab}
    if GLOBAL_STEER_TOKEN in dist:
        dist[GLOBAL_STEER_TOKEN] = 0.90
    return _normalize(dist)


@dataclass(frozen=True)
class Phase0Predictors:
    base: Predictor
    teacher: Predictor
    student: Predictor


def build_phase0_predictors(stub_mode: str, vocab: list[str]) -> Phase0Predictors:
    if stub_mode not in {"null", "global", "selective"}:
        raise ValueError(f"Unsupported phase0.stub_mode: {stub_mode}")

    def base(sample: Sample) -> Distribution:
        return _base_distribution(sample=sample, vocab=vocab)

    def teacher(sample: Sample) -> Distribution:
        return _teacher_distribution(sample=sample, vocab=vocab)

    def student(sample: Sample) -> Distribution:
        mode = sample.get("mode")
        if mode not in {"on", "off"}:
            raise ValueError(f"Sample.mode must be 'on' or 'off', got: {mode!r}")

        if stub_mode == "null":
            return base(sample)
        if stub_mode == "global":
            return _global_distribution(vocab)
        if mode == "on":
            return teacher(sample)
        return base(sample)

    return Phase0Predictors(base=base, teacher=teacher, student=student)

