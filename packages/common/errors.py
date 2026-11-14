"""
common.errors — Structured error hierarchy for QueryOpt Lab.

Design notes:
- All errors are subclasses of QueryOptError (base).
- Each error carries a machine-readable error_code for API responses.
- Human-readable messages are for the frontend; technical details go to the logger.
- Never raise bare Exception from within queryopt packages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Base ─────────────────────────────────────────────────────────────────────

@dataclass
class QueryOptError(Exception):
    """
    Base class for all QueryOpt Lab errors.

    Attributes:
        message: Human-readable description of what went wrong.
        error_code: Stable machine-readable identifier (e.g. "SQL_PARSE_ERROR").
        context: Optional dict of additional diagnostic information.
    """
    message: str
    error_code: str = "QUERYOPT_ERROR"
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }


# ── SQL and Parsing Errors ────────────────────────────────────────────────────

@dataclass
class SQLParseError(QueryOptError):
    """Raised when the SQL string cannot be parsed."""
    error_code: str = "SQL_PARSE_ERROR"


@dataclass
class UnsupportedQueryError(QueryOptError):
    """
    Raised when the SQL is syntactically valid but uses a feature
    not yet implemented in QueryOpt Lab.
    """
    error_code: str = "UNSUPPORTED_QUERY"
    feature: str = ""

    def __post_init__(self) -> None:
        if self.feature:
            self.context["unsupported_feature"] = self.feature


@dataclass
class SQLValidationError(QueryOptError):
    """
    Raised when the parsed SQL references unknown tables or columns,
    or contains type mismatches.
    """
    error_code: str = "SQL_VALIDATION_ERROR"


# ── Planning Errors ───────────────────────────────────────────────────────────

@dataclass
class PlanningError(QueryOptError):
    """Raised when the logical plan cannot be constructed from a valid AST."""
    error_code: str = "PLANNING_ERROR"


@dataclass
class OptimizationError(QueryOptError):
    """Raised when an optimizer fails to select or validate a plan."""
    error_code: str = "OPTIMIZATION_ERROR"
    optimizer_id: str = ""

    def __post_init__(self) -> None:
        if self.optimizer_id:
            self.context["optimizer_id"] = self.optimizer_id


@dataclass
class OptimizationCorrectnessFailure(QueryOptError):
    """
    CRITICAL: Raised when an optimized plan produces different results
    than the naive/reference plan. This indicates a semantic bug in an
    optimization rule or physical operator.

    See: Section 44 of the project specification.
    """
    error_code: str = "OPTIMIZATION_CORRECTNESS_FAILURE"
    query: str = ""
    optimizer_id: str = ""

    def __post_init__(self) -> None:
        if self.query:
            self.context["query"] = self.query
        if self.optimizer_id:
            self.context["optimizer_id"] = self.optimizer_id


# ── Execution Errors ──────────────────────────────────────────────────────────

@dataclass
class ExecutionError(QueryOptError):
    """Raised when query execution fails for a non-resource reason."""
    error_code: str = "EXECUTION_ERROR"


@dataclass
class ResourceLimitError(QueryOptError):
    """
    Raised when a query exceeds a configured resource limit.
    The frontend should display this as a controlled user-facing error,
    not as a crash.
    """
    error_code: str = "RESOURCE_LIMIT_EXCEEDED"
    limit_type: str = ""   # "time", "rows", "memory", "join_count", "plan_candidates"
    limit_value: Any = None
    actual_value: Any = None

    def __post_init__(self) -> None:
        self.context.update({
            "limit_type": self.limit_type,
            "limit_value": self.limit_value,
            "actual_value": self.actual_value,
        })


# ── Storage Errors ────────────────────────────────────────────────────────────

@dataclass
class StorageError(QueryOptError):
    """Raised for I/O or persistence failures in the storage layer."""
    error_code: str = "STORAGE_ERROR"


@dataclass
class TableNotFoundError(QueryOptError):
    """Raised when a query references a table that does not exist."""
    error_code: str = "TABLE_NOT_FOUND"
    table_name: str = ""

    def __post_init__(self) -> None:
        if self.table_name:
            self.context["table_name"] = self.table_name


@dataclass
class ColumnNotFoundError(SQLValidationError):
    """Raised when a query references a column that does not exist in the table."""
    error_code: str = "COLUMN_NOT_FOUND"
    column_name: str = ""
    table_name: str = ""

    def __post_init__(self) -> None:
        self.context.update({
            "column_name": self.column_name,
            "table_name": self.table_name,
        })


@dataclass
class TableAlreadyExistsError(QueryOptError):
    """Raised when CREATE TABLE is called for a table that already exists."""
    error_code: str = "TABLE_ALREADY_EXISTS"
    table_name: str = ""

    def __post_init__(self) -> None:
        if self.table_name:
            self.context["table_name"] = self.table_name


# ── Dataset and Experiment Errors ─────────────────────────────────────────────

@dataclass
class DatasetError(QueryOptError):
    """Raised for errors in dataset loading, generation, or validation."""
    error_code: str = "DATASET_ERROR"


@dataclass
class ExperimentConfigurationError(QueryOptError):
    """
    Raised when an experiment configuration YAML is invalid,
    references unknown optimizers, queries, or datasets,
    or violates reproducibility requirements.
    """
    error_code: str = "EXPERIMENT_CONFIG_ERROR"
    config_path: str = ""

    def __post_init__(self) -> None:
        if self.config_path:
            self.context["config_path"] = self.config_path


@dataclass
class ExperimentExecutionError(QueryOptError):
    """Raised when the experiment runner fails during benchmark execution."""
    error_code: str = "EXPERIMENT_EXECUTION_ERROR"
    experiment_id: str = ""

    def __post_init__(self) -> None:
        if self.experiment_id:
            self.context["experiment_id"] = self.experiment_id
