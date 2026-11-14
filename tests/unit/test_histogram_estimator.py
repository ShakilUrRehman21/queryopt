"""
Unit tests for histogram-based cardinality estimation.
"""
from __future__ import annotations

import pytest

from common.types import Column, DataType, Schema
from optimizer.annotator import CardinalityAnnotator
from parser.ast_types import BinaryOp, ColumnRef, Literal
from planner.nodes import LogicalFilter, LogicalScan
from query_stats.estimator import (
    HistogramCardinalityEstimator,
    UniformCardinalityEstimator,
)
from query_stats.histogram import (
    EquiDepthHistogram,
    EquiWidthHistogram,
    HistogramBucket,
)
from storage.statistics import ColumnStats, TableStatistics


class TestHistogramSerialization:

    def test_histogram_bucket_serialization(self):
        b = HistogramBucket(lower_bound=10.0, upper_bound=20.0, count=50, distinct_count=15)
        d = b.to_dict()
        b2 = HistogramBucket.from_dict(d)
        assert b2.lower_bound == 10.0
        assert b2.upper_bound == 20.0
        assert b2.count == 50
        assert b2.distinct_count == 15

    def test_equi_width_histogram_serialization(self):
        vals = [1.0, 2.0, 3.0, 10.0, 20.0, 50.0, 100.0]
        h = EquiWidthHistogram.build(vals, num_buckets=5)
        d = h.to_dict()
        h2 = EquiWidthHistogram.from_dict(d)
        assert len(h2.buckets) == len(h.buckets)
        assert h2.total_count == h.total_count
        assert h2.estimate_selectivity("<=", 25.0) == pytest.approx(h.estimate_selectivity("<=", 25.0))

    def test_equi_depth_histogram_serialization(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        h = EquiDepthHistogram.build(vals, num_buckets=3)
        d = h.to_dict()
        h2 = EquiDepthHistogram.from_dict(d)
        assert len(h2.buckets) == len(h.buckets)
        assert h2.total_count == h.total_count


class TestHistogramEstimator:

    def test_histogram_captures_skew(self):
        """
        On skewed data where 90% of values are in [0, 10] and 10% are in [10, 100],
        histogram estimate for <= 10 should be ~0.90 while uniform estimate is ~0.10.
        """
        # 90 values in [0, 10], 10 values in [10, 100]
        skewed_vals = [float(i % 10) for i in range(90)] + [float(i) for i in range(10, 20)]
        hist = EquiWidthHistogram.build(skewed_vals, num_buckets=10)

        col_stat = ColumnStats(
            column_name="skew_val",
            data_type=DataType.FLOAT,
            row_count=100,
            distinct_count=20,
            min_value=0.0,
            max_value=20.0,
        )
        col_stat.histogram = hist.to_dict()  # type: ignore[attr-defined]

        table_stats = {
            "test_tbl": TableStatistics("test_tbl", 100, {"skew_val": col_stat})
        }

        hist_estimator = HistogramCardinalityEstimator()
        unif_estimator = UniformCardinalityEstimator()

        pred = BinaryOp(
            ColumnRef(column="skew_val", table="test_tbl"),
            "<=",
            Literal(value=10.0, data_type=DataType.FLOAT),
        )

        hist_est = hist_estimator.estimate_filter(100.0, pred, table_stats)
        unif_est = unif_estimator.estimate_filter(100.0, pred, table_stats)

        # Histogram accurately predicts majority of rows (>= 0.85)
        assert hist_est.selectivity >= 0.85
        # Uniform predicts (10 - 0) / (20 - 0) = 0.50
        assert unif_est.selectivity <= 0.60

    def test_fallback_to_uniform_when_no_histogram(self):
        """Estimator seamlessly falls back to uniform when column has no histogram."""
        col_stat = ColumnStats(
            column_name="age",
            data_type=DataType.INTEGER,
            row_count=100,
            distinct_count=10,
            min_value=20,
            max_value=30,
        )
        table_stats = {
            "test_tbl": TableStatistics("test_tbl", 100, {"age": col_stat})
        }

        estimator = HistogramCardinalityEstimator()
        pred = BinaryOp(
            ColumnRef(column="age", table="test_tbl"),
            "=",
            Literal(value=25, data_type=DataType.INTEGER),
        )
        est = estimator.estimate_filter(100.0, pred, table_stats)
        # 1 / distinct_count = 1 / 10 = 0.10
        assert est.selectivity == pytest.approx(0.10)
        assert est.estimated_rows == pytest.approx(10.0)

    def test_annotator_integration(self):
        """CardinalityAnnotator works with HistogramCardinalityEstimator."""
        schema = Schema([Column("age", DataType.INTEGER)])
        scan = LogicalScan(schema=schema, table_name="test_tbl")
        filt = LogicalFilter(
            schema=schema,
            child=scan,
            predicate=BinaryOp(ColumnRef("age"), ">", Literal(50, DataType.INTEGER)),
        )

        annotator = CardinalityAnnotator(catalog=None, estimator=HistogramCardinalityEstimator())
        annotated = annotator.annotate(filt)
        assert annotated.estimated_rows is not None
        assert annotated.estimated_rows > 0
