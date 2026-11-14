"""
experiment.metrics — Metric collection and statistical aggregation for experiments.

Computes:
- Execution, planning, and end-to-end latency
- Estimated cost and physical plan properties
- Cardinality estimation accuracy via q-error: max(est / act, act / est)
- Statistical summaries: mean, median, p95, p99, stddev
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def calculate_q_error(estimated: float, actual: float) -> float:
    """
    Calculate q-error between estimated and actual cardinality.
    q-error = max(est / act, act / est), with 1.0 floor for actual/est == 0.
    """
    est = max(float(estimated), 1.0)
    act = max(float(actual), 1.0)
    ratio = est / act
    return max(ratio, 1.0 / ratio)


@dataclass
class QueryRunMetric:
    """Individual execution measurement for a single query repetition."""
    experiment_id: str
    query_id: str
    optimizer_id: str
    repetition: int
    is_warmup: bool = False
    planning_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    total_time_ms: float = 0.0
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0
    actual_rows: int = 0
    q_error: float = 1.0
    status: str = "SUCCESS"
    error_message: str | None = None
    rules_applied: list[str] = field(default_factory=list)
    alternatives_considered: int = 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rules_applied"] = ", ".join(self.rules_applied)
        return d


@dataclass
class MetricSummary:
    """Statistical summary across multiple repetitions."""
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    stddev: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    @classmethod
    def compute(cls, values: list[float]) -> MetricSummary:
        if not values:
            return cls()
        arr = sorted(values)
        n = len(arr)
        mean = sum(arr) / n
        variance = sum((x - mean) ** 2 for x in arr) / n if n > 1 else 0.0
        stddev = math.sqrt(variance)

        def percentile(p: float) -> float:
            idx = int(math.ceil(p * n)) - 1
            return arr[max(0, min(n - 1, idx))]

        return cls(
            count=n,
            mean=round(mean, 4),
            median=round(percentile(0.50), 4),
            stddev=round(stddev, 4),
            min_val=round(arr[0], 4),
            max_val=round(arr[-1], 4),
            p95=round(percentile(0.95), 4),
            p99=round(percentile(0.99), 4),
        )


@dataclass
class QuerySummary:
    """Summary of performance for a specific query + optimizer pair."""
    query_id: str
    optimizer_id: str
    execution_time_summary: MetricSummary
    planning_time_summary: MetricSummary
    total_time_summary: MetricSummary
    estimated_cost: float
    estimated_rows: float
    actual_rows: int
    q_error: float
    rules_applied: list[str] = field(default_factory=list)
    alternatives_considered: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "optimizer_id": self.optimizer_id,
            "exec_time_mean_ms": self.execution_time_summary.mean,
            "exec_time_median_ms": self.execution_time_summary.median,
            "exec_time_stddev": self.execution_time_summary.stddev,
            "exec_time_p95_ms": self.execution_time_summary.p95,
            "plan_time_mean_ms": self.planning_time_summary.mean,
            "total_time_mean_ms": self.total_time_summary.mean,
            "estimated_cost": round(self.estimated_cost, 4),
            "estimated_rows": round(self.estimated_rows, 2),
            "actual_rows": self.actual_rows,
            "q_error": round(self.q_error, 4),
            "rules_applied": ", ".join(self.rules_applied),
            "alternatives_considered": self.alternatives_considered,
        }
