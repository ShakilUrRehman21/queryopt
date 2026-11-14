"""
storage.statistics — Column statistics for cardinality estimation.

Statistics are computed on demand by TableHandle.collect_statistics().
They are used by the cost-based optimizer's cardinality estimator.

Statistics stored per column (Phase 1):
    - row_count
    - null_count
    - distinct_count (approximate via set)
    - min_value, max_value

Phase 2 extensions (not yet implemented):
    - Histograms (equal-width, equal-depth)
    - Most-common values (MCV) list
    - Correlation with other columns

Design notes:
- Statistics are NOT automatically updated on INSERT.
  ANALYZE must be called explicitly (or via the CLI: queryopt db analyze).
- distinct_count uses an exact set during collection (Phase 1).
  For large tables, replace with HyperLogLog.
- Statistics are serialized to JSON and stored in the catalog directory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.logging import get_logger
from common.types import Column, DataType

logger = get_logger(__name__)

STATS_FILENAME_TEMPLATE = "{table_name}_stats.json"


# ── Column Statistics ─────────────────────────────────────────────────────────

@dataclass
class ColumnStats:
    """
    Basic statistics for a single column.

    Used by the cardinality estimator to compute selectivity.
    """
    column_name: str
    data_type: DataType

    row_count: int = 0
    null_count: int = 0
    distinct_count: int = 0

    min_value: Any = None
    max_value: Any = None

    # Fraction of rows that are NULL: null_count / row_count
    @property
    def null_fraction(self) -> float:
        if self.row_count == 0:
            return 0.0
        return self.null_count / self.row_count

    # Non-null rows
    @property
    def non_null_count(self) -> int:
        return self.row_count - self.null_count

    def selectivity_equality(self, value: Any) -> float:
        """
        Estimate selectivity for a predicate: column = value.

        Uses uniform distribution assumption (Phase 1).
        Returns fraction of rows expected to match.

        ASSUMPTION: Values are uniformly distributed across distinct values.
        This is known to be inaccurate for skewed distributions (see RQ3).
        """
        if self.distinct_count == 0 or self.row_count == 0:
            return 0.0
        # Fraction of non-null rows: 1/distinct_count
        non_null_sel = 1.0 / self.distinct_count
        # Adjust for nulls
        return non_null_sel * (1.0 - self.null_fraction)

    def selectivity_range(self, low: Any, high: Any) -> float:
        """
        Estimate selectivity for: low <= column <= high.

        Uses uniform distribution over [min, max] range.
        For non-numeric types, defaults to 1/3 (uniform guess).

        ASSUMPTION: Uniform distribution over numeric range.
        """
        if self.data_type not in (DataType.INTEGER, DataType.FLOAT):
            return 1.0 / 3.0  # Heuristic for non-numeric ranges

        if self.min_value is None or self.max_value is None:
            return 1.0 / 3.0

        total_range = float(self.max_value) - float(self.min_value)
        if total_range <= 0:
            return 1.0 if float(low) <= float(self.min_value) <= float(high) else 0.0

        query_range = float(high) - float(low)
        sel = min(1.0, max(0.0, query_range / total_range))
        return sel * (1.0 - self.null_fraction)

    def to_dict(self) -> dict:
        return {
            "column_name": self.column_name,
            "data_type": self.data_type.value,
            "row_count": self.row_count,
            "null_count": self.null_count,
            "distinct_count": self.distinct_count,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnStats":
        return cls(
            column_name=d["column_name"],
            data_type=DataType(d["data_type"]),
            row_count=d.get("row_count", 0),
            null_count=d.get("null_count", 0),
            distinct_count=d.get("distinct_count", 0),
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
        )


# ── Builder (used during statistics collection scan) ──────────────────────────

class ColumnStatsBuilder:
    """Accumulates observations during a table scan to build ColumnStats."""

    def __init__(self, column: Column) -> None:
        self._column = column
        self._row_count = 0
        self._null_count = 0
        self._distinct_values: set = set()
        self._min_value: Any = None
        self._max_value: Any = None

    def observe(self, value: Any) -> None:
        """Record a single value during the scan."""
        self._row_count += 1
        if value is None:
            self._null_count += 1
            return

        # Only track distinct values exactly up to a limit to control memory
        # For Phase 1, exact tracking is fine for small tables.
        # Replace with HyperLogLog for large tables.
        if len(self._distinct_values) < 100_000:
            self._distinct_values.add(value)

        # Track min/max for numeric and text types
        if self._column.data_type in (DataType.INTEGER, DataType.FLOAT, DataType.TEXT,
                                       DataType.DATE, DataType.TIMESTAMP):
            if self._min_value is None or value < self._min_value:
                self._min_value = value
            if self._max_value is None or value > self._max_value:
                self._max_value = value

    def build(self) -> ColumnStats:
        return ColumnStats(
            column_name=self._column.name,
            data_type=self._column.data_type,
            row_count=self._row_count,
            null_count=self._null_count,
            distinct_count=len(self._distinct_values),
            min_value=self._min_value,
            max_value=self._max_value,
        )


# ── Table Statistics ──────────────────────────────────────────────────────────

@dataclass
class TableStatistics:
    """
    All statistics for a single table.

    Serialized to <db_dir>/<table_name>_stats.json.
    """
    table_name: str
    row_count: int
    column_stats: dict[str, ColumnStats] = field(default_factory=dict)

    @property
    def columns(self) -> dict[str, ColumnStats]:
        return self.column_stats

    def get_column(self, column_name: str) -> ColumnStats | None:
        return self.column_stats.get(column_name)

    def to_dict(self) -> dict:
        return {
            "table_name": self.table_name,
            "row_count": self.row_count,
            "columns": {name: cs.to_dict() for name, cs in self.column_stats.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableStatistics":
        return cls(
            table_name=d["table_name"],
            row_count=d["row_count"],
            column_stats={
                name: ColumnStats.from_dict(cs)
                for name, cs in d.get("columns", {}).items()
            },
        )

    def save(self, db_dir: Path) -> None:
        path = db_dir / STATS_FILENAME_TEMPLATE.format(table_name=self.table_name)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info(f"Saved statistics for '{self.table_name}' to {path}")

    @classmethod
    def load(cls, db_dir: Path, table_name: str) -> "TableStatistics | None":
        path = db_dir / STATS_FILENAME_TEMPLATE.format(table_name=table_name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Statistics file for '{table_name}' is corrupt; returning None")
            return None
