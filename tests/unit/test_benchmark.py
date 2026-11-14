"""
Unit tests for benchmark dataset generator and workloads.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.generator import (
    ColumnSpec,
    DataDistribution,
    DatasetGenerator,
    TableSpec,
)
from benchmark.workloads import (
    BenchmarkWorkload,
    create_workload,
    get_standard_tables,
)
from common.types import DataType


class TestDatasetGenerator:

    def test_reproducibility_with_seed(self):
        """Same seed must produce identical column values."""
        spec = ColumnSpec(
            name="col_val",
            data_type=DataType.FLOAT,
            distribution=DataDistribution.UNIFORM,
            min_val=10.0,
            max_val=100.0,
        )
        gen = DatasetGenerator(seed=12345)
        vals1 = gen.generate_column_values(spec, 50, np.random.default_rng(12345))
        vals2 = gen.generate_column_values(spec, 50, np.random.default_rng(12345))
        assert vals1 == vals2

    def test_sequential_distribution(self):
        """Sequential distribution increments monotonically."""
        spec = ColumnSpec(
            name="id",
            data_type=DataType.INTEGER,
            distribution=DataDistribution.SEQUENTIAL,
            min_val=100,
        )
        gen = DatasetGenerator()
        vals = gen.generate_column_values(spec, 5, np.random.default_rng(42))
        assert vals == [100, 101, 102, 103, 104]

    def test_normal_distribution_bounds(self):
        """Normal distribution values are clipped to [min_val, max_val]."""
        spec = ColumnSpec(
            name="rating",
            data_type=DataType.FLOAT,
            distribution=DataDistribution.NORMAL,
            min_val=1.0,
            max_val=5.0,
            mean=3.0,
            std=1.0,
        )
        gen = DatasetGenerator()
        vals = gen.generate_column_values(spec, 200, np.random.default_rng(42))
        assert len(vals) == 200
        assert all(1.0 <= v <= 5.0 for v in vals)

    def test_zipfian_skew(self):
        """Zipfian distribution exhibits frequency concentration at lower ranks."""
        spec = ColumnSpec(
            name="hot_key",
            data_type=DataType.INTEGER,
            distribution=DataDistribution.ZIPFIAN,
            min_val=1,
            max_val=10,
            zipf_param=2.0,
        )
        gen = DatasetGenerator()
        vals = gen.generate_column_values(spec, 1000, np.random.default_rng(42))
        assert len(vals) == 1000
        # Rank 1 should be most frequent
        count_1 = vals.count(1)
        count_10 = vals.count(10)
        assert count_1 > count_10

    def test_null_injection(self):
        """Null fraction injects approximately expected proportion of None values."""
        spec = ColumnSpec(
            name="opt_col",
            data_type=DataType.INTEGER,
            distribution=DataDistribution.UNIFORM,
            null_fraction=0.30,
        )
        gen = DatasetGenerator()
        vals = gen.generate_column_values(spec, 1000, np.random.default_rng(42))
        null_count = sum(1 for v in vals if v is None)
        # 30% +- 7% tolerance for sample of 1000
        assert 230 <= null_count <= 370

    def test_generate_table(self):
        """Table generation produces rows matching schema."""
        spec = TableSpec(
            name="test_tbl",
            row_count=20,
            columns=[
                ColumnSpec("id", DataType.INTEGER, DataDistribution.SEQUENTIAL),
                ColumnSpec("val", DataType.FLOAT, DataDistribution.UNIFORM, min_val=0.0, max_val=10.0),
            ],
        )
        gen = DatasetGenerator()
        rows = gen.generate_table(spec)
        assert len(rows) == 20
        assert "id" in rows[0]
        assert "val" in rows[0]
        assert rows[0]["id"] == 1
        assert rows[19]["id"] == 20


class TestBenchmarkWorkloads:

    @pytest.mark.parametrize("wid", ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9"])
    def test_all_workloads_instantiate(self, wid: str):
        """All standard workloads W1..W7 can be created with valid queries and tables."""
        wl = create_workload(wid, scale=1)
        assert isinstance(wl, BenchmarkWorkload)
        assert wl.workload_id == wid
        assert len(wl.queries) > 0
        assert len(wl.table_specs) > 0
        for q in wl.queries:
            assert q.sql.strip().upper().startswith("SELECT")

    def test_workload_scale_factor(self):
        """Scaling workload increases row counts proportionally."""
        wl1 = create_workload("W1", scale=1)
        wl2 = create_workload("W1", scale=5)
        assert wl2.table_specs[0].row_count == wl1.table_specs[0].row_count * 5

    def test_invalid_workload_raises(self):
        with pytest.raises(ValueError, match="Unknown workload ID"):
            create_workload("W99")
