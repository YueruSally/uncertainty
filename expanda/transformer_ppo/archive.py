"""Reward-only nondominated archive and exact normalized 3-D HV."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


Point = Tuple[float, float, float]


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated(points: Iterable[Sequence[float]]) -> list[Point]:
    unique = sorted({tuple(float(x) for x in point) for point in points})
    return [p for p in unique if not any(_dominates(q, p) for q in unique if q != p)]


def exact_hv_3d(points: Iterable[Sequence[float]], reference: Sequence[float]) -> float:
    """Exact dominated volume for minimization points in three dimensions.

    The sweep decomposes the x axis into slabs.  In each slab it computes the
    union area of origin-anchored rectangles in the y-z plane.
    """
    ref = tuple(float(x) for x in reference)
    if len(ref) != 3 or any(not np.isfinite(x) for x in ref):
        raise ValueError("finite three-dimensional reference required")
    clipped = nondominated(
        p for p in points
        if len(p) == 3 and all(np.isfinite(p)) and all(p[i] < ref[i] for i in range(3))
    )
    if not clipped:
        return 0.0
    xs = sorted({p[0] for p in clipped} | {ref[0]})
    volume = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        active = [(p[1], p[2]) for p in clipped if p[0] <= left]
        if not active or right <= left:
            continue
        ys = sorted({p[0] for p in active} | {ref[1]})
        area = 0.0
        for low_y, high_y in zip(ys[:-1], ys[1:]):
            zs = [z for y, z in active if y <= low_y]
            if zs and high_y > low_y:
                area += (high_y - low_y) * max(0.0, ref[2] - min(zs))
        volume += (right - left) * area
    return float(volume)


@dataclass(frozen=True)
class ObjectiveNormalizer:
    minimum: Point
    maximum: Point
    reference: Point = (1.1, 1.1, 1.1)
    divisor: Point = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if any(not np.isfinite(x) for x in (
                *self.minimum, *self.maximum, *self.reference, *self.divisor)):
            raise ValueError("normalization values must be finite")
        if any(hi <= lo for lo, hi in zip(self.minimum, self.maximum)):
            raise ValueError("each normalization maximum must exceed its minimum")
        if any(value <= 0 for value in self.divisor):
            raise ValueError("objective divisors must be positive")

    def transform(self, objectives: Sequence[float]) -> Point:
        if len(objectives) != 3 or any(not np.isfinite(x) for x in objectives):
            raise ValueError("finite three-objective vector required")
        return tuple(
            (float(value) / divisor - lo) / (hi - lo)
            for value, divisor, lo, hi in zip(
                objectives, self.divisor, self.minimum, self.maximum)
        )  # type: ignore[return-value]

    def with_divisor(self, divisor: Sequence[float]):
        if len(divisor) != 3:
            raise ValueError("three objective divisors required")
        return ObjectiveNormalizer(
            self.minimum, self.maximum, self.reference,
            tuple(float(x) for x in divisor))  # type: ignore[arg-type]


@dataclass(frozen=True)
class RewardConfig:
    positive_hv_scale: float = 0.01
    no_change_penalty: float = 0.01
    infeasible_penalty: float = 0.05
    repair_failure_penalty: float = 0.10
    violation_improvement_weight: float = 0.025
    reward_clip: float = 1.0

    def validate(self) -> None:
        values = (
            self.positive_hv_scale,
            self.no_change_penalty,
            self.infeasible_penalty,
            self.repair_failure_penalty,
            self.reward_clip,
        )
        if any(x <= 0 for x in values) or self.violation_improvement_weight < 0:
            raise ValueError("reward scales and penalties must be positive")


class RewardArchive:
    """Archive used only for reward; it never participates in NSGA-II selection."""

    def __init__(self, normalizer: ObjectiveNormalizer):
        self.normalizer = normalizer
        self._points: list[Point] = []
        self._fingerprints: set[str] = set()

    @property
    def points(self) -> tuple[Point, ...]:
        return tuple(self._points)

    @property
    def hypervolume(self) -> float:
        return exact_hv_3d(self._points, self.normalizer.reference)

    def observe(self, objectives: Sequence[float], feasible: bool, fingerprint: str) -> float:
        before = self.hypervolume
        if not feasible or fingerprint in self._fingerprints:
            return 0.0
        try:
            point = self.normalizer.transform(objectives)
        except ValueError:
            return 0.0
        self._fingerprints.add(fingerprint)
        self._points = nondominated([*self._points, point])
        return max(0.0, self.hypervolume - before)

    def summary(self) -> dict:
        if not self._points:
            return {"archive_size": 0, "archive_hv": 0.0,
                    "archive_min": [0.0] * 3, "archive_max": [0.0] * 3}
        values = np.asarray(self._points, dtype=float)
        return {"archive_size": len(self._points), "archive_hv": self.hypervolume,
                "archive_min": values.min(axis=0).tolist(),
                "archive_max": values.max(axis=0).tolist()}


def mutation_reward(row: dict, delta_hv: float, config: RewardConfig) -> float:
    config.validate()
    if row.get("repair_success") is False or row.get("mutation_reverted"):
        value = -config.repair_failure_penalty
    elif not row.get("decision_changed"):
        value = -config.no_change_penalty
    elif not row.get("feasible_after"):
        old = float(row.get("violation_before") or 0.0)
        new = float(row.get("violation_after") or old)
        improvement = np.clip(old - new, -1.0, 1.0)
        value = -config.infeasible_penalty + config.violation_improvement_weight * improvement
    elif delta_hv > 0:
        value = delta_hv / config.positive_hv_scale
    else:
        value = 0.0
    return float(np.clip(value, -config.reward_clip, config.reward_clip))
