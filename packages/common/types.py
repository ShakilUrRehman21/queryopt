"""
common.types — Core domain types shared across all QueryOpt Lab packages.

Design notes:
- All types are immutable (frozen dataclasses or Enum) wherever possible.
- Row is a tuple for memory efficiency during iteration.
- DataType is an Enum to prevent string-comparison bugs.
- Schema is ordered: column position matters for unnamed INSERT and SELECT *.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, NamedTuple


# ── Data Types ───────────────────────────────────────────────────────────────

class DataType(Enum):
    """Supported column data types in the QueryOpt Lab engine."""
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    TEXT = "TEXT"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"
    NULL = "NULL"  # Internal: for NULL literals and unresolved types

    def is_numeric(self) -> bool:
        return self in (DataType.INTEGER, DataType.FLOAT)

    def is_comparable_to(self, other: DataType) -> bool:
        """Return True if two types can be compared without explicit casting."""
        if self == other:
            return True
        # INTEGER and FLOAT are interoperable
        if {self, other} <= {DataType.INTEGER, DataType.FLOAT}:
            return True
        # NULL is comparable to anything (IS NULL / IS NOT NULL)
        if DataType.NULL in (self, other):
            return True
        return False

    def python_type(self) -> type:
        """Return the Python type that represents values of this DataType."""
        _map: dict[DataType, type] = {
            DataType.INTEGER: int,
            DataType.FLOAT: float,
            DataType.BOOLEAN: bool,
            DataType.TEXT: str,
            DataType.DATE: str,        # stored as ISO date string
            DataType.TIMESTAMP: str,   # stored as ISO datetime string
            DataType.NULL: type(None),
        }
        return _map[self]


# ── Column and Schema ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Column:
    """Metadata for a single column in a table schema."""
    name: str
    data_type: DataType
    nullable: bool = True
    primary_key: bool = False
    default_value: Any = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Column name cannot be empty.")
        if not isinstance(self.data_type, DataType):
            raise TypeError(f"data_type must be DataType enum, got {type(self.data_type)}")


@dataclass(frozen=True)
class Schema:
    """
    Ordered sequence of columns defining a table's structure.

    Column order is significant: it determines positional INSERT semantics
    and the order of fields in a Row tuple.
    """
    columns: tuple[Column, ...]

    def __init__(self, columns: list[Column] | tuple[Column, ...]) -> None:
        object.__setattr__(self, "columns", tuple(columns))
        self._validate()

    def _validate(self) -> None:
        names = [col.name for col in self.columns]
        if len(names) != len(set(names)):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate column names in schema: {duplicates}")

    def column_names(self) -> list[str]:
        return [col.name for col in self.columns]

    def column_index(self, name: str) -> int:
        """Return 0-based index of column by name. Raises KeyError if not found."""
        for i, col in enumerate(self.columns):
            if col.name == name:
                return i
        raise KeyError(f"Column '{name}' not found in schema. Available: {self.column_names()}")

    def get_column(self, name: str) -> Column:
        idx = self.column_index(name)
        return self.columns[idx]

    def has_column(self, name: str) -> bool:
        return any(col.name == name for col in self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def __iter__(self):
        return iter(self.columns)


# ── Row ──────────────────────────────────────────────────────────────────────

# A Row is a plain tuple of Python values.
# Position corresponds to Schema.columns position.
# Using tuple for immutability and low memory overhead.
Row = tuple[Any, ...]


def make_row(schema: Schema, values: dict[str, Any]) -> Row:
    """
    Build a Row tuple from a mapping of column_name → value.

    Columns not present in values receive None (NULL) if nullable,
    or the column's default_value if specified.
    """
    result: list[Any] = []
    for col in schema.columns:
        if col.name in values:
            result.append(values[col.name])
        elif col.default_value is not None:
            result.append(col.default_value)
        elif col.nullable:
            result.append(None)
        else:
            raise ValueError(f"Non-nullable column '{col.name}' has no value and no default.")
    return tuple(result)


def row_to_dict(schema: Schema, row: Row) -> dict[str, Any]:
    """Convert a Row tuple back to a column_name → value mapping."""
    return dict(zip(schema.column_names(), row))


# ── QueryResult ───────────────────────────────────────────────────────────────

class ExecutionStatus(Enum):
    """Status of a query execution."""
    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    RESOURCE_LIMIT = auto()


@dataclass
class QueryResult:
    """
    The complete result of executing a query through the QueryOpt Lab engine.

    This is the top-level return type from VolcanoExecutor.execute().
    """
    status: ExecutionStatus
    rows: list[Row]
    schema: Schema | None
    error_message: str | None = None
    error_type: str | None = None

    # Timing
    planning_time_ms: float = 0.0
    execution_time_ms: float = 0.0

    # Row statistics
    rows_returned: int = 0
    rows_scanned: int = 0

    # Memory
    peak_memory_bytes: int = 0

    # Operator-level statistics (operator_id → OperatorStats)
    operator_stats: dict[str, "OperatorStats"] = field(default_factory=dict)

    # Provenance reported by the optimizer (estimator used, feedback hits,
    # LLM model/rationale, ...). Empty for DDL/DML.
    optimizer_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_time_ms(self) -> float:
        return self.planning_time_ms + self.execution_time_ms

    @property
    def succeeded(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS


@dataclass
class OperatorStats:
    """Execution statistics for a single physical operator node."""
    operator_id: str
    operator_type: str
    estimated_rows: float = 0.0
    actual_rows: int = 0
    execution_time_ms: float = 0.0
    memory_bytes: int = 0

    @property
    def cardinality_error_ratio(self) -> float | None:
        """
        Ratio of actual to estimated rows. Values >> 1.0 indicate under-estimation;
        values << 1.0 indicate over-estimation. Returns None if estimate is 0.
        """
        if self.estimated_rows == 0:
            return None
        return self.actual_rows / self.estimated_rows


# ── Join Types ───────────────────────────────────────────────────────────────

class JoinType(Enum):
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"   # Not implemented in Phase 1; here for completeness
    FULL = "FULL JOIN"     # Not implemented in Phase 1


# ── Sort Order ───────────────────────────────────────────────────────────────

class SortDirection(Enum):
    ASC = "ASC"
    DESC = "DESC"


@dataclass(frozen=True)
class SortKey:
    """A single sort key with direction."""
    column_name: str
    direction: SortDirection = SortDirection.ASC
