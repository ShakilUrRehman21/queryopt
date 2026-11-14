"""
query_stats.noisy — Controlled cardinality-error injection (Research Gap 4, RQ2).

Wraps any CardinalityEstimator and distorts its filter/join estimates by a
configurable q-error, turning estimation quality into an *independent
variable* instead of something only measured after the fact.

Modes
-----
under   divide every estimate by q            (systematic under-estimation)
over    multiply every estimate by q          (systematic over-estimation)
random  multiply by q^u, u ∈ [-1, 1]          (deterministic per predicate + seed)

The random factor is derived from a hash of the canonical predicate text, so
the same query sees the same distortion in every repetition and process —
experiments stay reproducible.
"""
from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

from query_stats.estimator import CardinalityEstimate, CardinalityEstimator

if TYPE_CHECKING:
    from parser.ast_types import Expr
    from storage.statistics import TableStatistics


class NoisyCardinalityEstimator(CardinalityEstimator):
    def __init__(
        self,
        base: CardinalityEstimator,
        q_error: float = 10.0,
        mode: str = "random",
        seed: int = 0,
        apply_to: tuple[str, ...] = ("filter", "join"),
    ) -> None:
        if q_error < 1.0:
            raise ValueError("q_error must be ≥ 1")
        if mode not in ("under", "over", "random"):
            raise ValueError("mode must be 'under', 'over' or 'random'")
        self.base = base
        self.q_error = float(q_error)
        self.mode = mode
        self.seed = int(seed)
        self.apply_to = set(apply_to)

    def _factor(self, expr: Expr | None) -> float:
        if self.mode == "under":
            return 1.0 / self.q_error
        if self.mode == "over":
            return self.q_error
        from optimizer.signature import canonical_expr

        h = zlib.crc32(f"{self.seed}:{canonical_expr(expr)}".encode())
        u = (h / 0xFFFFFFFF) * 2.0 - 1.0
        return self.q_error**u

    def _distort(self, est: CardinalityEstimate, expr: Expr | None) -> CardinalityEstimate:
        f = self._factor(expr)
        return CardinalityEstimate(
            estimated_rows=max(1.0, est.estimated_rows * f),
            selectivity=min(1.0, est.selectivity * f),
        )

    def estimate_filter(self, input_rows, predicate, table_stats=None):  # type: ignore[override]
        est = self.base.estimate_filter(input_rows, predicate, table_stats)
        return self._distort(est, predicate) if "filter" in self.apply_to else est

    def estimate_join(self, left_rows, right_rows, condition, table_stats=None):  # type: ignore[override]
        est = self.base.estimate_join(left_rows, right_rows, condition, table_stats)
        return self._distort(est, condition) if "join" in self.apply_to else est

    def estimate_aggregate(self, input_rows, group_by, table_stats=None):  # type: ignore[override]
        return self.base.estimate_aggregate(input_rows, group_by, table_stats)
