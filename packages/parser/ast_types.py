"""
parser.ast_types — Internal AST node types produced by the parser.

These are QueryOpt Lab's own representations, independent of SQLGlot.
The planner only ever sees these types — never raw SQLGlot nodes.

Supported in Phase 1:
    SELECT (projections, FROM, WHERE, JOIN, GROUP BY, ORDER BY, LIMIT)
    CREATE TABLE
    INSERT INTO ... VALUES

Not supported (will raise UnsupportedQueryError):
    DROP, UPDATE, DELETE, subqueries, CTEs, window functions, UNION, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.types import DataType, JoinType, SortDirection


# ── Expression Nodes (used in WHERE, projections, JOIN conditions) ─────────────

@dataclass(frozen=True)
class ColumnRef:
    """Reference to a column, optionally qualified with a table alias."""
    column: str
    table: str | None = None  # alias or table name

    def __str__(self) -> str:
        return f"{self.table}.{self.column}" if self.table else self.column


@dataclass(frozen=True)
class Literal:
    """A constant literal value."""
    value: Any
    data_type: DataType


@dataclass(frozen=True)
class BinaryOp:
    """A binary operation: left OP right."""
    left: "Expr"
    op: str          # "=", "!=", "<", "<=", ">", ">=", "AND", "OR", "+", "-", "*", "/"
    right: "Expr"


@dataclass(frozen=True)
class UnaryOp:
    """A unary operation: OP expr."""
    op: str          # "NOT", "-", "IS NULL", "IS NOT NULL"
    expr: "Expr"


@dataclass(frozen=True)
class FunctionCall:
    """An aggregate or scalar function call: FUNC(args...)."""
    name: str        # e.g., "COUNT", "SUM", "AVG", "MIN", "MAX"
    args: tuple["Expr", ...]
    distinct: bool = False


@dataclass(frozen=True)
class Star:
    """SELECT * — all columns."""
    table: str | None = None  # optional table qualifier


# Expr is the union of all expression node types.
Expr = ColumnRef | Literal | BinaryOp | UnaryOp | FunctionCall | Star


# ── Projection Item ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProjectionItem:
    """
    A single item in a SELECT clause.

    Examples:
        SELECT id          → ProjectionItem(expr=ColumnRef("id"), alias=None)
        SELECT id AS user_id → ProjectionItem(expr=ColumnRef("id"), alias="user_id")
        SELECT COUNT(*)    → ProjectionItem(expr=FunctionCall("COUNT",[Star()]), alias=None)
        SELECT *           → ProjectionItem(expr=Star(), alias=None)
    """
    expr: Expr
    alias: str | None = None

    @property
    def output_name(self) -> str | None:
        """The name this projection will have in the result schema."""
        if self.alias:
            return self.alias
        if isinstance(self.expr, ColumnRef):
            return self.expr.column
        if isinstance(self.expr, FunctionCall):
            return self.expr.name.lower()
        if isinstance(self.expr, Star):
            return None  # expands to multiple columns
        return None


# ── Table Source ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TableSource:
    """A table reference in the FROM clause."""
    table_name: str
    alias: str | None = None

    @property
    def effective_alias(self) -> str:
        return self.alias if self.alias else self.table_name


# ── Join Clause ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class JoinClause:
    """A JOIN specification."""
    table: TableSource
    join_type: JoinType
    condition: Expr  # ON condition


# ── Sort Item ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SortItem:
    """A single ORDER BY item."""
    expr: Expr
    direction: SortDirection = SortDirection.ASC


# ── Statement Nodes ───────────────────────────────────────────────────────────

@dataclass
class SelectStatement:
    """Parsed representation of a SELECT query."""
    projections: list[ProjectionItem]
    from_table: TableSource
    joins: list[JoinClause] = field(default_factory=list)
    where: Expr | None = None
    group_by: list[Expr] = field(default_factory=list)
    having: Expr | None = None
    order_by: list[SortItem] = field(default_factory=list)
    limit: int | None = None
    offset: int | None = None
    distinct: bool = False


@dataclass
class CreateTableStatement:
    """Parsed representation of CREATE TABLE."""
    table_name: str
    columns: list[ColumnDefinition]
    if_not_exists: bool = False


@dataclass
class ColumnDefinition:
    """Column definition in a CREATE TABLE statement."""
    name: str
    data_type: DataType
    nullable: bool = True
    primary_key: bool = False
    default_value: Any = None


@dataclass
class InsertStatement:
    """Parsed representation of INSERT INTO ... VALUES (...)."""
    table_name: str
    columns: list[str] | None      # None = positional (no column list specified)
    values_rows: list[list[Any]]   # Each inner list is one row


# The union of all supported statement types
Statement = SelectStatement | CreateTableStatement | InsertStatement


@dataclass
class ParseResult:
    """The output of the parser for a single SQL string."""
    sql: str
    statement: Statement
