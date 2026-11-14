"""
statistics.estimator — Cardinality and selectivity estimation.

Provides the foundational mathematical models used by the Cost-Based Optimizer
(CBO) to estimate how many rows each operator in a query plan will produce.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from common.types import DataType
from parser.ast_types import (
    BinaryOp,
    ColumnRef,
    Expr,
    FunctionCall,
    Literal,
    UnaryOp,
)

if TYPE_CHECKING:
    from storage.catalog import Catalog
    from storage.statistics import ColumnStats, TableStatistics


@dataclass
class CardinalityEstimate:
    """Estimated row count and associated selectivity factor."""
    estimated_rows: float
    selectivity: float = 1.0


class CardinalityEstimator(ABC):
    """Abstract base class for cardinality and selectivity estimators."""

    @abstractmethod
    def estimate_filter(
        self,
        input_rows: float,
        predicate: Expr,
        table_stats: dict[str, TableStatistics] | None = None,
    ) -> CardinalityEstimate:
        """Estimate output rows from a filter operator."""
        ...

    @abstractmethod
    def estimate_join(
        self,
        left_rows: float,
        right_rows: float,
        condition: Expr,
        table_stats: dict[str, TableStatistics] | None = None,
    ) -> CardinalityEstimate:
        """Estimate output rows from a join operator."""
        ...

    @abstractmethod
    def estimate_aggregate(
        self,
        input_rows: float,
        group_by: list[Expr],
        table_stats: dict[str, TableStatistics] | None = None,
    ) -> CardinalityEstimate:
        """Estimate output rows from an aggregate operator."""
        ...


class UniformCardinalityEstimator(CardinalityEstimator):
    """
    Standard textbook cardinality estimator (System R style).
    Assumes uniform attribute value distribution and attribute independence.
    """

    def __init__(self, default_range_selectivity: float = 0.33, default_eq_selectivity: float = 0.1) -> None:
        self.default_range_sel = default_range_selectivity
        self.default_eq_sel = default_eq_selectivity

    def estimate_filter(
        self,
        input_rows: float,
        predicate: Expr,
        table_stats: dict[str, TableStatistics] | None = None,
    ) -> CardinalityEstimate:
        sel = self._compute_selectivity(predicate, table_stats)
        sel = max(0.0, min(1.0, sel))
        est_rows = max(1.0, input_rows * sel) if input_rows > 0 else 0.0
        return CardinalityEstimate(estimated_rows=est_rows, selectivity=sel)

    def _compute_selectivity(
        self,
        expr: Expr,
        table_stats: dict[str, TableStatistics] | None,
    ) -> float:
        match expr:
            case Literal():
                return 1.0 if bool(expr.value) else 0.0

            case UnaryOp():
                op = expr.op.upper()
                if op == "NOT":
                    return 1.0 - self._compute_selectivity(expr.expr, table_stats)
                if op == "IS NULL":
                    col_stats = self._find_column_stats(expr.expr, table_stats)
                    if col_stats and col_stats.row_count > 0:
                        return col_stats.null_count / col_stats.row_count
                    return 0.05
                if op == "IS NOT NULL":
                    return 1.0 - self._compute_selectivity(
                        UnaryOp(op="IS NULL", expr=expr.expr), table_stats
                    )
                return 1.0

            case BinaryOp():
                op = expr.op.upper()
                if op == "AND":
                    # Attribute independence assumption: P(A AND B) = P(A) * P(B)
                    sel_l = self._compute_selectivity(expr.left, table_stats)
                    sel_r = self._compute_selectivity(expr.right, table_stats)
                    return sel_l * sel_r

                if op == "OR":
                    # P(A OR B) = P(A) + P(B) - P(A AND B)
                    sel_l = self._compute_selectivity(expr.left, table_stats)
                    sel_r = self._compute_selectivity(expr.right, table_stats)
                    return sel_l + sel_r - (sel_l * sel_r)

                # Comparison with column and literal
                col_ref: ColumnRef | None = None
                lit: Literal | None = None

                if isinstance(expr.left, ColumnRef) and isinstance(expr.right, Literal):
                    col_ref = expr.left
                    lit = expr.right
                elif isinstance(expr.right, ColumnRef) and isinstance(expr.left, Literal):
                    col_ref = expr.right
                    lit = expr.left
                    # Invert op if operands are flipped
                    flip_map = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
                    op = flip_map.get(op, op)

                if col_ref is not None and lit is not None:
                    col_stats = self._find_column_stats(col_ref, table_stats)
                    return self._estimate_comparison(op, col_stats, lit.value)

                # Fallback for complex binary expressions
                if op == "=":
                    return self.default_eq_sel
                if op in ("<", "<=", ">", ">="):
                    return self.default_range_sel
                if op == "!=":
                    return 1.0 - self.default_eq_sel
                return 1.0

            case _:
                return 1.0

    def _estimate_comparison(
        self,
        op: str,
        col_stats: ColumnStats | None,
        val: Any,
    ) -> float:
        if col_stats is None:
            if op == "=":
                return self.default_eq_sel
            if op == "!=":
                return 1.0 - self.default_eq_sel
            return self.default_range_sel

        distinct = max(col_stats.distinct_count, 1)

        if op == "=":
            # 1 / distinct_count (uniform distribution assumption)
            return 1.0 / distinct

        if op == "!=":
            return 1.0 - (1.0 / distinct)

        # Range comparisons: interpolate if numeric min/max available
        if op in ("<", "<=", ">", ">=") and col_stats.min_value is not None and col_stats.max_value is not None:
            try:
                min_v = float(col_stats.min_value)
                max_v = float(col_stats.max_value)
                cur_v = float(val)

                if min_v >= max_v:
                    return self.default_range_sel

                if op in ("<", "<="):
                    if cur_v <= min_v:
                        return 0.01
                    if cur_v >= max_v:
                        return 1.0
                    return (cur_v - min_v) / (max_v - min_v)
                else:
                    if cur_v <= min_v:
                        return 1.0
                    if cur_v >= max_v:
                        return 0.01
                    return (max_v - cur_v) / (max_v - min_v)
            except (ValueError, TypeError):
                pass

        return self.default_range_sel

    def _find_column_stats(
        self,
        expr: Expr,
        table_stats: dict[str, TableStatistics] | None,
    ) -> ColumnStats | None:
        if not table_stats or not isinstance(expr, ColumnRef):
            return None

        col_name = expr.column.lower()
        tbl_name = expr.table.lower() if expr.table else None

        if tbl_name and tbl_name in table_stats:
            return table_stats[tbl_name].get_column(col_name)

        # Search across all available table stats if table wasn't qualified
        for t_stat in table_stats.values():
            c_stat = t_stat.get_column(col_name)
            if c_stat is not None:
                return c_stat
        return None

    def estimate_join(
        self,
        left_rows: float,
        right_rows: float,
        condition: Expr,
        table_stats: dict[str, TableStatistics] | None = None,
    ) -> CardinalityEstimate:
        """
        Estimate join cardinality using System R equi-join formula:
          est_rows = (left_rows * right_rows) / max(distinct(col_left), distinct(col_right))
        """
        if isinstance(condition, BinaryOp) and condition.op == "=":
            col_l = self._find_column_stats(condition.left, table_stats)
            col_r = self._find_column_stats(condition.right, table_stats)

            distinct_l = col_l.distinct_count if col_l else 10
            distinct_r = col_r.distinct_count if col_r else 10
            denom = max(distinct_l, distinct_r, 1)

            cartesian = left_rows * right_rows
            est_rows = cartesian / denom
            sel = 1.0 / denom if denom > 0 else 0.1
            return CardinalityEstimate(estimated_rows=max(1.0, est_rows), selectivity=sel)

        # Cross join or non-equi join fallback
        cartesian = left_rows * right_rows
        est_rows = cartesian * 0.33
        return CardinalityEstimate(estimated_rows=max(1.0, est_rows), selectivity=0.33)

    def estimate_aggregate(
        self,
        input_rows: float,
        group_by: list[Expr],
        table_stats: dict[str, TableStatistics] | None = None,
    ) -> CardinalityEstimate:
        """
        Estimate aggregate output rows:
        - If no GROUP BY: exactly 1 row (scalar aggregate).
        - If GROUP BY: product of distinct counts of grouping keys, capped at input_rows.
        """
        if not group_by:
            return CardinalityEstimate(estimated_rows=1.0, selectivity=1.0 / max(input_rows, 1.0))

        prod_distinct = 1.0
        for gb_expr in group_by:
            c_stat = self._find_column_stats(gb_expr, table_stats)
            distinct_val = c_stat.distinct_count if c_stat else 5
            prod_distinct *= max(distinct_val, 1)

        est_rows = min(input_rows, prod_distinct)
        return CardinalityEstimate(
            estimated_rows=max(1.0, est_rows),
            selectivity=est_rows / max(input_rows, 1.0),
        )


class HistogramCardinalityEstimator(UniformCardinalityEstimator):
    """
    Histogram-based cardinality and selectivity estimator.

    Uses column-level histograms (EquiWidthHistogram or EquiDepthHistogram)
    when available to compute non-uniform, data-driven selectivity estimates.
    Gracefully falls back to textbook uniform estimates if a column has no
    histogram available.
    """

    def __init__(
        self,
        histograms: dict[str, Any] | None = None,
        default_range_selectivity: float = 0.33,
        default_eq_selectivity: float = 0.1,
    ) -> None:
        super().__init__(
            default_range_selectivity=default_range_selectivity,
            default_eq_selectivity=default_eq_selectivity,
        )
        self.histograms = histograms or {}

    def _estimate_comparison(
        self,
        op: str,
        col_stats: ColumnStats | None,
        val: Any,
    ) -> float:
        from query_stats.histogram import EquiDepthHistogram, EquiWidthHistogram

        # 1. Check if an explicit histogram was registered for this column
        hist = None
        if col_stats and col_stats.column_name in self.histograms:
            hist = self.histograms[col_stats.column_name]
        elif col_stats and getattr(col_stats, "histogram", None) is not None:
            raw_h = col_stats.histogram
            if isinstance(raw_h, (EquiWidthHistogram, EquiDepthHistogram)):
                hist = raw_h
            elif isinstance(raw_h, dict):
                h_type = raw_h.get("type", "equi_width")
                if h_type == "equi_depth":
                    hist = EquiDepthHistogram.from_dict(raw_h)
                else:
                    hist = EquiWidthHistogram.from_dict(raw_h)

        if hist is not None and isinstance(val, (int, float)):
            return hist.estimate_selectivity(op, float(val))

        # 2. Fall back to uniform estimation
        return super()._estimate_comparison(op, col_stats, val)
