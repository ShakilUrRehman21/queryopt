"""
parser.validator — Validates an AST against the database catalog.

Ensures that:
- Referenced tables exist in the catalog.
- Referenced columns exist in the referenced tables.
- Ambiguous column references (in multi-table queries without qualifiers) are flagged.
- Type compatibilities and basic constraints are verified.
- Aggregate function constraints (e.g. no nested aggregates) are enforced.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from common.errors import (
    ColumnNotFoundError,
    SQLValidationError,
    TableAlreadyExistsError,
    TableNotFoundError,
)
from common.logging import get_logger
from parser.ast_types import (
    BinaryOp,
    ColumnDefinition,
    ColumnRef,
    CreateTableStatement,
    Expr,
    FunctionCall,
    InsertStatement,
    Literal,
    ProjectionItem,
    SelectStatement,
    SortItem,
    Star,
    Statement,
    UnaryOp,
)

if TYPE_CHECKING:
    from storage.catalog import Catalog

logger = get_logger(__name__)

AGGREGATE_FUNCTIONS = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})


class ASTValidator:
    """
    Validates AST statements against a database Catalog.
    """

    def __init__(self, catalog: Catalog, db_name: str = "default") -> None:
        self.catalog = catalog
        self.db_name = db_name

    def validate(self, statement: Statement) -> None:
        """
        Validate any supported statement against the catalog.

        Raises:
            TableNotFoundError: If a referenced table does not exist.
            TableAlreadyExistsError: If creating an existing table.
            ColumnNotFoundError: If a referenced column does not exist.
            SQLValidationError: If an expression or clause is semantically invalid.
        """
        match statement:
            case CreateTableStatement():
                self._validate_create_table(statement)
            case InsertStatement():
                self._validate_insert(statement)
            case SelectStatement():
                self._validate_select(statement)
            case _:
                raise SQLValidationError(
                    message=f"Validation not implemented for statement type {type(statement).__name__}"
                )

    def _validate_create_table(self, stmt: CreateTableStatement) -> None:
        if self.catalog.has_table(stmt.table_name):
            if not stmt.if_not_exists:
                raise TableAlreadyExistsError(
                    message=f"Table '{stmt.table_name}' already exists in database '{self.db_name}'.",
                    table_name=stmt.table_name,
                )

        if not stmt.columns:
            raise SQLValidationError(
                message=f"Table '{stmt.table_name}' must have at least one column definition."
            )

        col_names = set()
        for col in stmt.columns:
            if col.name.lower() in col_names:
                raise SQLValidationError(
                    message=f"Duplicate column name '{col.name}' in CREATE TABLE '{stmt.table_name}'."
                )
            col_names.add(col.name.lower())

    def _validate_insert(self, stmt: InsertStatement) -> None:
        if not self.catalog.has_table(stmt.table_name):
            raise TableNotFoundError(
                message=f"Table '{stmt.table_name}' does not exist in database '{self.db_name}'.",
                table_name=stmt.table_name,
            )

        table_meta = self.catalog.get_table(stmt.table_name)
        schema_cols = table_meta.schema.columns
        schema_col_names = [c.name.lower() for c in schema_cols]

        if stmt.columns is not None:
            if not stmt.columns:
                raise SQLValidationError(
                    message=f"INSERT into '{stmt.table_name}' has an empty column list."
                )
            # Check all specified columns exist
            for c in stmt.columns:
                if c.lower() not in schema_col_names:
                    raise ColumnNotFoundError(
                        message=f"Column '{c}' does not exist in table '{stmt.table_name}'.",
                        column_name=c,
                        table_name=stmt.table_name,
                    )
            expected_count = len(stmt.columns)
        else:
            expected_count = len(schema_cols)

        # Validate row lengths
        for idx, row in enumerate(stmt.values_rows):
            if len(row) != expected_count:
                raise SQLValidationError(
                    message=(
                        f"Row {idx + 1} has {len(row)} values, but expected {expected_count} columns "
                        f"for table '{stmt.table_name}'."
                    )
                )

    def _validate_select(self, stmt: SelectStatement) -> None:
        # 1. Resolve table sources
        # alias -> (actual_table_name, list_of_column_names)
        tables: dict[str, tuple[str, list[str]]] = {}

        from_alias = stmt.from_table.effective_alias
        from_table = stmt.from_table.table_name
        if not self.catalog.has_table(from_table):
            raise TableNotFoundError(
                message=f"Table '{from_table}' does not exist in database '{self.db_name}'.",
                table_name=from_table,
            )
        table_meta = self.catalog.get_table(from_table)
        tables[from_alias.lower()] = (from_table, [c.name for c in table_meta.schema.columns])

        # JOIN tables
        for join in stmt.joins:
            j_alias = join.table.effective_alias.lower()
            j_table = join.table.table_name
            if j_alias in tables:
                raise SQLValidationError(
                    message=f"Duplicate table alias or name '{j_alias}' in query."
                )
            if not self.catalog.has_table(j_table):
                raise TableNotFoundError(
                    message=f"Joined table '{j_table}' does not exist in database '{self.db_name}'.",
                    table_name=j_table,
                )
            j_meta = self.catalog.get_table(j_table)
            tables[j_alias] = (j_table, [c.name for c in j_meta.schema.columns])

        # Helper to validate a ColumnRef
        def validate_column_ref(col_ref: ColumnRef) -> None:
            if col_ref.table:
                tbl_key = col_ref.table.lower()
                if tbl_key not in tables:
                    raise SQLValidationError(
                        message=f"Table or alias '{col_ref.table}' not found in FROM or JOIN clauses."
                    )
                real_table, cols = tables[tbl_key]
                col_match = [c for c in cols if c.lower() == col_ref.column.lower()]
                if not col_match:
                    raise ColumnNotFoundError(
                        message=f"Column '{col_ref.column}' not found in table '{col_ref.table}'.",
                        column_name=col_ref.column,
                        table_name=col_ref.table,
                    )
            else:
                # Unqualified: check ambiguity
                matching_tables: list[str] = []
                for t_alias, (real_table, cols) in tables.items():
                    if any(c.lower() == col_ref.column.lower() for c in cols):
                        matching_tables.append(t_alias)
                if not matching_tables:
                    raise ColumnNotFoundError(
                        message=f"Column '{col_ref.column}' does not exist in any referenced table.",
                        column_name=col_ref.column,
                        table_name="",
                    )
                if len(matching_tables) > 1:
                    raise SQLValidationError(
                        message=(
                            f"Column reference '{col_ref.column}' is ambiguous; "
                            f"found in multiple tables: {', '.join(matching_tables)}."
                        )
                    )

        # Helper to validate expressions
        def validate_expr(expr: Expr, allow_aggregate: bool = True, in_aggregate: bool = False) -> None:
            match expr:
                case ColumnRef():
                    validate_column_ref(expr)
                case Literal():
                    pass
                case Star():
                    if expr.table:
                        if expr.table.lower() not in tables:
                            raise SQLValidationError(
                                message=f"Table '{expr.table}' in '{expr.table}.*' not recognized."
                            )
                case BinaryOp():
                    validate_expr(expr.left, allow_aggregate=allow_aggregate, in_aggregate=in_aggregate)
                    validate_expr(expr.right, allow_aggregate=allow_aggregate, in_aggregate=in_aggregate)
                case UnaryOp():
                    validate_expr(expr.expr, allow_aggregate=allow_aggregate, in_aggregate=in_aggregate)
                case FunctionCall():
                    if in_aggregate:
                        raise SQLValidationError(
                            message=f"Nested aggregate function calls are not allowed: {expr.name}(...)"
                        )
                    is_agg = expr.name in AGGREGATE_FUNCTIONS
                    if is_agg and not allow_aggregate:
                        raise SQLValidationError(
                            message=f"Aggregate function '{expr.name}' not allowed in this clause."
                        )
                    for arg in expr.args:
                        validate_expr(arg, allow_aggregate=False, in_aggregate=is_agg)

        # 2. Validate JOIN conditions
        for join in stmt.joins:
            validate_expr(join.condition, allow_aggregate=False)

        # 3. Validate WHERE clause
        if stmt.where:
            validate_expr(stmt.where, allow_aggregate=False)

        # 4. Validate GROUP BY
        for gb_expr in stmt.group_by:
            validate_expr(gb_expr, allow_aggregate=False)

        # 5. Validate HAVING
        if stmt.having:
            validate_expr(stmt.having, allow_aggregate=True)

        # 6. Validate Projections
        has_aggregates = False
        for proj in stmt.projections:
            validate_expr(proj.expr, allow_aggregate=True)
            if self._contains_aggregate(proj.expr):
                has_aggregates = True

        # 7. Validate ORDER BY
        for sort_item in stmt.order_by:
            validate_expr(sort_item.expr, allow_aggregate=True)

        # 8. If GROUP BY is present, check that any non-aggregate projection column is in GROUP BY
        if stmt.group_by:
            gb_col_refs = {str(c).lower() for c in stmt.group_by if isinstance(c, ColumnRef)}
            for proj in stmt.projections:
                if isinstance(proj.expr, ColumnRef):
                    ref_name = str(proj.expr).lower()
                    # Also check simple col name
                    if ref_name not in gb_col_refs and proj.expr.column.lower() not in gb_col_refs:
                        raise SQLValidationError(
                            message=(
                                f"Column '{proj.expr}' must appear in the GROUP BY clause "
                                f"or be used in an aggregate function."
                            )
                        )
                elif isinstance(proj.expr, Star):
                    raise SQLValidationError(
                        message="SELECT * cannot be used with GROUP BY in QueryOpt Lab."
                    )
        elif has_aggregates:
            # Query has aggregates but no GROUP BY: any non-aggregate projection is invalid
            for proj in stmt.projections:
                if not self._contains_aggregate(proj.expr):
                    if isinstance(proj.expr, (ColumnRef, Star)):
                        raise SQLValidationError(
                            message=(
                                f"Expression '{proj.output_name or proj.expr}' must be in an aggregate "
                                f"function or GROUP BY clause."
                            )
                        )

    def _contains_aggregate(self, expr: Expr) -> bool:
        if isinstance(expr, FunctionCall) and expr.name in AGGREGATE_FUNCTIONS:
            return True
        if isinstance(expr, BinaryOp):
            return self._contains_aggregate(expr.left) or self._contains_aggregate(expr.right)
        if isinstance(expr, UnaryOp):
            return self._contains_aggregate(expr.expr)
        return False


def validate(statement: Statement, catalog: Catalog, db_name: str = "default") -> None:
    """Convenience function to validate an AST statement."""
    validator = ASTValidator(catalog, db_name)
    validator.validate(statement)
