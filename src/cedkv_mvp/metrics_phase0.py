from __future__ import annotations

import math
from typing import Mapping


def to_topk_sparse(dist: Mapping[str, float], top_k: int) -> dict[str, float]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ranked = sorted(dist.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return {token: max(value, 0.0) for token, value in ranked}


def kl_student_to_base_union_other(
    student_sparse: Mapping[str, float],
    base_sparse: Mapping[str, float],
    eps: float = 1e-12,
) -> float:
    """Compute KL(student || base) on union(student_keys, base_keys) + OTHER bucket."""
    if eps <= 0:
        raise ValueError("eps must be > 0")

    student = {token: max(value, 0.0) for token, value in student_sparse.items()}
    base = {token: max(value, 0.0) for token, value in base_sparse.items()}

    student_mass = sum(student.values())
    base_mass = sum(base.values())

    if student_mass > 1.0:
        student = {token: value / student_mass for token, value in student.items()}
        student_mass = 1.0
    if base_mass > 1.0:
        base = {token: value / base_mass for token, value in base.items()}
        base_mass = 1.0

    union_tokens = set(student).union(base)
    student_other = max(0.0, 1.0 - student_mass)
    base_other = max(0.0, 1.0 - base_mass)

    student_values = [student.get(token, 0.0) for token in union_tokens] + [student_other]
    base_values = [base.get(token, 0.0) for token in union_tokens] + [base_other]

    student_eps = [value + eps for value in student_values]
    base_eps = [value + eps for value in base_values]

    student_norm = sum(student_eps)
    base_norm = sum(base_eps)
    student_prob = [value / student_norm for value in student_eps]
    base_prob = [value / base_norm for value in base_eps]

    return sum(p * math.log(p / q) for p, q in zip(student_prob, base_prob))


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    rank = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[rank]

