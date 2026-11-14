"""
statistics.histogram — Equi-width and equi-depth histogram implementations.

Provides non-uniform selectivity estimation for skewed distributions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class HistogramBucket:
    """A single bucket in a histogram."""
    lower_bound: float
    upper_bound: float
    count: int
    distinct_count: int = 1

    def contains(self, val: float) -> bool:
        return self.lower_bound <= val <= self.upper_bound

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "count": self.count,
            "distinct_count": self.distinct_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HistogramBucket":
        return cls(
            lower_bound=float(d["lower_bound"]),
            upper_bound=float(d["upper_bound"]),
            count=int(d["count"]),
            distinct_count=int(d.get("distinct_count", 1)),
        )


class EquiWidthHistogram:
    """
    Equi-width histogram dividing [min, max] into B equal-width buckets.
    """

    def __init__(self, buckets: list[HistogramBucket], total_count: int) -> None:
        self.buckets = buckets
        self.total_count = total_count

    @classmethod
    def build(cls, values: Sequence[float], num_buckets: int = 10) -> "EquiWidthHistogram":
        valid = [float(v) for v in values if v is not None and not math.isnan(v)]
        if not valid:
            return cls([], 0)

        min_val = min(valid)
        max_val = max(valid)
        total = len(valid)

        if min_val == max_val or num_buckets <= 1:
            return cls(
                [HistogramBucket(min_val, max_val, total, len(set(valid)))],
                total,
            )

        width = (max_val - min_val) / num_buckets
        buckets = [
            HistogramBucket(
                lower_bound=min_val + i * width,
                upper_bound=min_val + (i + 1) * width,
                count=0,
                distinct_count=0,
            )
            for i in range(num_buckets)
        ]
        # Ensure upper bound of last bucket covers max_val
        buckets[-1].upper_bound = max_val

        bucket_sets: list[set[float]] = [set() for _ in range(num_buckets)]
        for v in valid:
            idx = min(int((v - min_val) / width), num_buckets - 1)
            buckets[idx].count += 1
            bucket_sets[idx].add(v)

        for i, b in enumerate(buckets):
            b.distinct_count = len(bucket_sets[i])

        return cls(buckets, total)

    def estimate_selectivity(self, op: str, val: float) -> float:
        """Estimate selectivity for a comparison against a numeric value."""
        if not self.buckets or self.total_count == 0:
            return 0.33

        op = op.upper()
        if op == "=":
            for b in self.buckets:
                if b.contains(val):
                    d = max(b.distinct_count, 1)
                    return (b.count / d) / self.total_count
            return 0.001

        if op in ("<", "<="):
            accum_rows = 0.0
            for b in self.buckets:
                if val >= b.upper_bound:
                    accum_rows += b.count
                elif val <= b.lower_bound:
                    break
                else:
                    # Partial bucket interpolation
                    ratio = (val - b.lower_bound) / max(b.upper_bound - b.lower_bound, 1e-9)
                    accum_rows += b.count * max(0.0, min(1.0, ratio))
                    break
            return max(0.001, min(1.0, accum_rows / self.total_count))

        if op in (">", ">="):
            return 1.0 - self.estimate_selectivity("<=", val)

        return 0.33

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "equi_width",
            "buckets": [b.to_dict() for b in self.buckets],
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EquiWidthHistogram":
        buckets = [HistogramBucket.from_dict(b) for b in d.get("buckets", [])]
        return cls(buckets, int(d.get("total_count", 0)))


class EquiDepthHistogram:
    """
    Equi-depth histogram where each bucket holds approximately equal row counts.
    """

    def __init__(self, buckets: list[HistogramBucket], total_count: int) -> None:
        self.buckets = buckets
        self.total_count = total_count

    @classmethod
    def build(cls, values: Sequence[float], num_buckets: int = 10) -> "EquiDepthHistogram":
        valid = sorted([float(v) for v in values if v is not None and not math.isnan(v)])
        if not valid:
            return cls([], 0)

        total = len(valid)
        num_buckets = min(num_buckets, total)
        bucket_size = total / num_buckets

        buckets: list[HistogramBucket] = []
        for i in range(num_buckets):
            start_idx = int(i * bucket_size)
            end_idx = int((i + 1) * bucket_size) - 1
            if i == num_buckets - 1:
                end_idx = total - 1

            b_vals = valid[start_idx : end_idx + 1]
            buckets.append(
                HistogramBucket(
                    lower_bound=b_vals[0],
                    upper_bound=b_vals[-1],
                    count=len(b_vals),
                    distinct_count=len(set(b_vals)),
                )
            )

        return cls(buckets, total)

    def estimate_selectivity(self, op: str, val: float) -> float:
        if not self.buckets or self.total_count == 0:
            return 0.33

        op = op.upper()
        if op == "=":
            for b in self.buckets:
                if b.contains(val):
                    d = max(b.distinct_count, 1)
                    return (b.count / d) / self.total_count
            return 0.001

        if op in ("<", "<="):
            accum_rows = 0.0
            for b in self.buckets:
                if val >= b.upper_bound:
                    accum_rows += b.count
                elif val <= b.lower_bound:
                    break
                else:
                    width = b.upper_bound - b.lower_bound
                    ratio = (val - b.lower_bound) / width if width > 0 else 0.5
                    accum_rows += b.count * max(0.0, min(1.0, ratio))
                    break
            return max(0.001, min(1.0, accum_rows / self.total_count))

        if op in (">", ">="):
            return 1.0 - self.estimate_selectivity("<=", val)

        return 0.33

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "equi_depth",
            "buckets": [b.to_dict() for b in self.buckets],
            "total_count": self.total_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EquiDepthHistogram":
        buckets = [HistogramBucket.from_dict(b) for b in d.get("buckets", [])]
        return cls(buckets, int(d.get("total_count", 0)))
