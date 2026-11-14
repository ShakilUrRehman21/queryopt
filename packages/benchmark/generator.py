"""
benchmark.generator — Synthetic dataset generator with controllable distributions.

Supports:
- UNIFORM: Equal probability across values.
- NORMAL: Gaussian distribution with specified mean and stddev.
- ZIPFIAN: Power-law / skewed distribution for modeling realistic skew.
- SEQUENTIAL: Monotonically increasing integer sequences (e.g. primary keys).
- CORRELATED: Values derived from another column plus Gaussian noise, to
  break the attribute-independence assumption on purpose (W8).

All generation is fully reproducible via random seeds (ADR-006).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from common.logging import get_logger
from common.types import Column, DataType, Schema

if TYPE_CHECKING:
    from storage.table import Database

logger = get_logger(__name__)


class DataDistribution(str, Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    ZIPFIAN = "zipfian"
    SEQUENTIAL = "sequential"
    CORRELATED = "correlated"


@dataclass
class ColumnSpec:
    """Specification for generating values of a single column."""
    name: str
    data_type: DataType = DataType.INTEGER
    distribution: DataDistribution = DataDistribution.UNIFORM
    min_val: float = 1.0
    max_val: float = 1000.0
    mean: float | None = None
    std: float | None = None
    zipf_param: float = 1.5   # Parameter a > 1 for Zipfian distribution
    distinct_values: int | None = None
    null_fraction: float = 0.0
    text_prefix: str = "val_"
    # CORRELATED only: name of the (earlier) column this one follows, the
    # strength of the relationship (+1 = same direction, -1 = inverse), and
    # the Gaussian noise added, as a fraction of this column's range.
    correlated_with: str | None = None
    correlation_sign: float = 1.0
    correlation_noise: float = 0.05


@dataclass
class TableSpec:
    """Specification for generating a relational table."""
    name: str
    row_count: int
    columns: list[ColumnSpec] = field(default_factory=list)

    def to_schema(self) -> Schema:
        return Schema([
            Column(
                name=c.name,
                data_type=c.data_type,
                nullable=(c.null_fraction > 0.0),
            )
            for c in self.columns
        ])


class DatasetGenerator:
    """
    Generates deterministic, synthetic datasets for query optimization benchmarks.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def generate_column_values(
        self,
        spec: ColumnSpec,
        row_count: int,
        rng: np.random.Generator,
        base_values: list[Any] | None = None,
    ) -> list[Any]:
        """Generate a list of row_count values for the given ColumnSpec."""
        if row_count <= 0:
            return []

        # 1. Generate base numeric values according to distribution
        match spec.distribution:
            case DataDistribution.SEQUENTIAL:
                start = int(spec.min_val)
                raw_nums = np.arange(start, start + row_count, dtype=np.float64)

            case DataDistribution.UNIFORM:
                raw_nums = rng.uniform(spec.min_val, spec.max_val, size=row_count)

            case DataDistribution.NORMAL:
                mean = spec.mean if spec.mean is not None else (spec.min_val + spec.max_val) / 2.0
                std = spec.std if spec.std is not None else max(1.0, (spec.max_val - spec.min_val) / 6.0)
                raw_nums = rng.normal(mean, std, size=row_count)
                # Clip to [min_val, max_val]
                raw_nums = np.clip(raw_nums, spec.min_val, spec.max_val)

            case DataDistribution.ZIPFIAN:
                # Zipf distribution over 1..K distinct ranks
                k = spec.distinct_values or max(2, int(spec.max_val - spec.min_val + 1))
                a = max(1.01, spec.zipf_param)
                # Sample ranks from zipf, then shift to [min_val, max_val]
                ranks = rng.zipf(a, size=row_count)
                ranks = np.clip(ranks, 1, k)
                # Map rank to value range
                span = spec.max_val - spec.min_val
                raw_nums = spec.min_val + ((ranks - 1) / max(k - 1, 1)) * span

            case DataDistribution.CORRELATED:
                if base_values is None:
                    raise ValueError(
                        f"Column '{spec.name}' is CORRELATED but its base column "
                        f"'{spec.correlated_with}' was not generated before it."
                    )
                base = np.array(
                    [np.nan if v is None else float(v) for v in base_values], dtype=np.float64
                )
                b_min, b_max = np.nanmin(base), np.nanmax(base)
                unit = (base - b_min) / max(b_max - b_min, 1e-9)
                if spec.correlation_sign < 0:
                    unit = 1.0 - unit
                unit = np.nan_to_num(unit, nan=0.5)
                unit = unit + rng.normal(0.0, spec.correlation_noise, size=row_count)
                span = spec.max_val - spec.min_val
                raw_nums = np.clip(spec.min_val + unit * span, spec.min_val, spec.max_val)

            case _:
                raw_nums = rng.uniform(spec.min_val, spec.max_val, size=row_count)

        # 2. If distinct_values constraint is specified, discretize
        if spec.distinct_values is not None and spec.distinct_values > 0:
            k = spec.distinct_values
            discrete_levels = np.linspace(spec.min_val, spec.max_val, k)
            indices = np.clip(
                np.round(((raw_nums - spec.min_val) / max(spec.max_val - spec.min_val, 1e-9)) * (k - 1)).astype(int),
                0,
                k - 1,
            )
            raw_nums = discrete_levels[indices]

        # 3. Convert to target DataType
        result: list[Any] = []
        for val in raw_nums:
            if spec.data_type == DataType.INTEGER:
                result.append(int(round(val)))
            elif spec.data_type == DataType.FLOAT:
                result.append(float(round(val, 4)))
            elif spec.data_type == DataType.TEXT:
                result.append(f"{spec.text_prefix}{int(round(val))}")
            elif spec.data_type == DataType.BOOLEAN:
                result.append(bool(val > (spec.min_val + spec.max_val) / 2.0))
            else:
                result.append(int(round(val)))

        # 4. Inject NULLs if requested
        if spec.null_fraction > 0.0:
            null_mask = rng.random(size=row_count) < spec.null_fraction
            for i, is_null in enumerate(null_mask):
                if is_null:
                    result[i] = None

        return result

    def generate_table(self, spec: TableSpec) -> list[dict[str, Any]]:
        """Generate all rows for a table as a list of dictionaries."""
        rng = np.random.default_rng(self.seed)
        col_arrays: dict[str, list[Any]] = {}

        for col_spec in spec.columns:
            # Seed uniquely per column for independent random streams.
            # zlib.crc32 is stable across processes; the built-in hash() of a
            # str is salted per interpreter run (PYTHONHASHSEED), which made
            # "seeded" datasets differ between runs.
            col_seed = (self.seed + zlib.crc32(col_spec.name.encode("utf-8"))) % (2**32)
            col_rng = np.random.default_rng(col_seed)
            base_values = (
                col_arrays.get(col_spec.correlated_with) if col_spec.correlated_with else None
            )
            col_arrays[col_spec.name] = self.generate_column_values(
                col_spec, spec.row_count, col_rng, base_values=base_values
            )

        # Zip into list of row dicts
        rows: list[dict[str, Any]] = []
        for i in range(spec.row_count):
            row_dict = {col_name: col_arrays[col_name][i] for col_name in col_arrays}
            rows.append(row_dict)

        return rows

    def populate_database(
        self,
        db: Database,
        specs: list[TableSpec],
        collect_statistics: bool = True,
    ) -> None:
        """
        Create and populate tables directly in a Database instance,
        and optionally collect and persist column statistics.
        """
        for spec in specs:
            schema = spec.to_schema()
            table_handle = db.create_table(spec.name, schema)
            rows = self.generate_table(spec)

            logger.info(f"Populating table '{spec.name}' with {len(rows)} rows...")
            for row in rows:
                table_handle.insert(row)

            if collect_statistics:
                stats = table_handle.collect_statistics()
                stats.save(db.db_dir)
                logger.info(f"Collected statistics for table '{spec.name}' (rows={stats.row_count}).")
