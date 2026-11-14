"""
planner.builder — Constructs a LogicalPlanTree from a parsed SelectStatement.

Given a validated SelectStatement and a database Catalog, this module builds
the canonical (naive) logical operator tree:

          LogicalLimit (if LIMIT/OFFSET)
               │
          LogicalSort (if ORDER BY)
               │
          LogicalProject (projections)
               │
          LogicalFilter (if HAVING)
               │
          LogicalAggregate (if GROUP BY or aggregates present)
               │
          LogicalFilter (if WHERE)
               │
          LogicalJoin (if JOINs)
               │
          LogicalScan (FROM table)

This initial tree serves as the input to the optimizer, which will reorder,
push down, and transform operators.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from common.errors import PlanningError
from common.logging import get_logger
from common.types import Column, DataType, Schema
from parser.ast_types import (
    BinaryOp,
    ColumnRef,
    Expr,
    FunctionCall,
    Literal,
    ProjectionItem,
    SelectStatement,
    Star,
    UnaryOp,
)
from planner.nodes import (
    LogicalAggregate,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalNode,
    LogicalProject,
    LogicalScan,
    LogicalSort,
)

if TYPE_CHECKING:
    from storage.catalog import Catalog

logger = get_logger(__name__)

AGGREGATE_NAMES = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})


def _infer_expr_type(expr: Expr, schema: Schema) -> DataType:
    """Infer the resulting DataType of an expression given the input schema."""
    match expr:
        case Literal():
            return expr.data_type
        case ColumnRef():
            # Match by unqualified or qualified name
            for col in schema.columns:
                if col.name.lower() == expr.column.lower():
                    return col.data_type
            return DataType.TEXT
        case BinaryOp():
            if expr.op in ("=", "!=", "<", "<=", ">", ">=", "AND", "OR"):
                return DataType.BOOLEAN
            if expr.op in ("+", "-", "*", "/"):
                left_t = _infer_expr_type(expr.left, schema)
                right_t = _infer_expr_type(expr.right, schema)
                if left_t == DataType.FLOAT or right_t == DataType.FLOAT or expr.op == "/":
                    return DataType.FLOAT
                return DataType.INTEGER
            return DataType.TEXT
        case UnaryOp():
            if expr.op in ("NOT", "IS NULL", "IS NOT NULL"):
                return DataType.BOOLEAN
            return _infer_expr_type(expr.expr, schema)
        case FunctionCall():
            fname = expr.name.upper()
            if fname == "COUNT":
                return DataType.INTEGER
            if fname == "AVG":
                return DataType.FLOAT
            if fname in ("SUM", "MIN", "MAX") and expr.args:
                return _infer_expr_type(expr.args[0], schema)
            return DataType.TEXT
        case _:
            return DataType.TEXT


def _contains_aggregate(expr: Expr) -> bool:
    """Check whether an expression contains an aggregate function call."""
    if isinstance(expr, FunctionCall) and expr.name.upper() in AGGREGATE_NAMES:
        return True
    if isinstance(expr, BinaryOp):
        return _contains_aggregate(expr.left) or _contains_aggregate(expr.right)
    if isinstance(expr, UnaryOp):
        return _contains_aggregate(expr.expr)
    return False


class LogicalPlanBuilder:
    """Constructs a LogicalPlan from an AST and catalog."""

    def __init__(self, catalog: Catalog, db_name: str = "default") -> None:
        self.catalog = catalog
        self.db_name = db_name

    def build(self, stmt: SelectStatement) -> LogicalNode:
        """
        Build the canonical logical plan for a SELECT statement.
        """
        # 1. Build leaf scan for the FROM table
        from_table = stmt.from_table.table_name
        table_meta = self.catalog.get_table(from_table)
        current_node: LogicalNode = LogicalScan(
            schema=table_meta.schema,
            table_name=from_table,
            alias=stmt.from_table.alias,
            projection_columns=[c.name for c in table_meta.schema.columns],
        )

        # 2. Join additional tables
        for join in stmt.joins:
            j_table = join.table.table_name
            j_meta = self.catalog.get_table(j_table)
            j_scan = LogicalScan(
                schema=j_meta.schema,
                table_name=j_table,
                alias=join.table.alias,
                projection_columns=[c.name for c in j_meta.schema.columns],
            )
            # Combine schemas: prefix column names if needed or keep both
            combined_cols = list(current_node.schema.columns)
            # Check for name collisions and prefix with alias/table if collided
            existing_names = {c.name.lower() for c in combined_cols}
            right_alias = join.table.effective_alias
            for c in j_meta.schema.columns:
                col_name = c.name
                if col_name.lower() in existing_names:
                    col_name = f"{right_alias}.{col_name}"
                combined_cols.append(
                    Column(
                        name=col_name,
                        data_type=c.data_type,
                        nullable=c.nullable or (join.join_type.value == "LEFT"),
                        default_value=c.default_value,
                    )
                )

            current_node = LogicalJoin(
                schema=Schema(combined_cols),
                left=current_node,
                right=j_scan,
                join_type=join.join_type,
                condition=join.condition,
            )

        # 3. Apply WHERE clause
        if stmt.where is not None:
            current_node = LogicalFilter(
                schema=current_node.schema,
                child=current_node,
                predicate=stmt.where,
            )

        # 4. Handle Aggregation (GROUP BY or Aggregate functions in projections)
        has_aggregates = any(_contains_aggregate(p.expr) for p in stmt.projections)
        if stmt.group_by or has_aggregates:
            agg_items: list[ProjectionItem] = []
            agg_cols: list[Column] = []

            # Add group by keys to output schema
            for idx, gb_expr in enumerate(stmt.group_by):
                col_name = str(gb_expr)
                if isinstance(gb_expr, ColumnRef):
                    col_name = gb_expr.column
                dt = _infer_expr_type(gb_expr, current_node.schema)
                agg_cols.append(Column(name=col_name, data_type=dt))

            # Add aggregate expressions
            for proj in stmt.projections:
                if _contains_aggregate(proj.expr):
                    out_name = proj.output_name or f"agg_{len(agg_items)}"
                    dt = _infer_expr_type(proj.expr, current_node.schema)
                    agg_items.append(proj)
                    agg_cols.append(Column(name=out_name, data_type=dt))

            current_node = LogicalAggregate(
                schema=Schema(agg_cols),
                child=current_node,
                group_by=stmt.group_by,
                aggregates=agg_items,
            )

        # 5. Apply HAVING clause
        if stmt.having is not None:
            current_node = LogicalFilter(
                schema=current_node.schema,
                child=current_node,
                predicate=stmt.having,
            )

        # 6. Apply Projections
        expanded_projections = self._expand_projections(stmt.projections, current_node.schema)
        if has_aggregates:
            replaced_projections = []
            for p in expanded_projections:
                if _contains_aggregate(p.expr):
                    col_name = p.output_name or f"agg_{p.expr.name.lower()}"
                    replaced_projections.append(
                        ProjectionItem(expr=ColumnRef(column=col_name), alias=p.alias)
                    )
                else:
                    replaced_projections.append(p)
            expanded_projections = replaced_projections

        proj_cols: list[Column] = []
        seen_names: set[str] = set()
        for p in expanded_projections:
            out_name = p.output_name or str(p.expr)
            if out_name.lower() in seen_names:
                if isinstance(p.expr, ColumnRef) and p.expr.table:
                    out_name = f"{p.expr.table}.{out_name}"
                count = 2
                base = out_name
                while out_name.lower() in seen_names:
                    out_name = f"{base}_{count}"
                    count += 1
            seen_names.add(out_name.lower())
            dt = _infer_expr_type(p.expr, current_node.schema)
            proj_cols.append(Column(name=out_name, data_type=dt))

        # Check if projection is a no-op (i.e. identical columns in identical order without renaming)
        is_noop = (
            len(expanded_projections) == len(current_node.schema.columns)
            and all(
                isinstance(p.expr, ColumnRef)
                and p.expr.column.lower() == c.name.lower()
                and (p.alias is None or p.alias.lower() == c.name.lower())
                for p, c in zip(expanded_projections, current_node.schema.columns)
            )
        )
        if not is_noop:
            current_node = LogicalProject(
                schema=Schema(proj_cols),
                child=current_node,
                projections=expanded_projections,
            )

        # 7. Apply ORDER BY
        if stmt.order_by:
            current_node = LogicalSort(
                schema=current_node.schema,
                child=current_node,
                sort_items=stmt.order_by,
            )

        # 8. Apply LIMIT / OFFSET
        if stmt.limit is not None or stmt.offset is not None:
            current_node = LogicalLimit(
                schema=current_node.schema,
                child=current_node,
                limit=stmt.limit if stmt.limit is not None else 0,
                offset=stmt.offset,
            )

        return current_node

    def _expand_projections(
        self, projections: list[ProjectionItem], input_schema: Schema
    ) -> list[ProjectionItem]:
        """Expands '*' and 'table.*' into explicit ColumnRef projections."""
        result: list[ProjectionItem] = []
        for item in projections:
            if isinstance(item.expr, Star):
                tbl_prefix = item.expr.table
                for col in input_schema.columns:
                    if tbl_prefix:
                        # Only include columns matching prefix or prefix.col
                        if col.name.startswith(f"{tbl_prefix}.") or "." not in col.name:
                            result.append(
                                ProjectionItem(
                                    expr=ColumnRef(column=col.name.split(".")[-1], table=tbl_prefix),
                                    alias=col.name.split(".")[-1],
                                )
                            )
                    else:
                        base_col = col.name.split(".")[-1]
                        result.append(
                            ProjectionItem(
                                expr=ColumnRef(column=base_col),
                                alias=base_col,
                            )
                        )
            else:
                result.append(item)
        return result


def build_logical_plan(
    stmt: SelectStatement, catalog: Catalog, db_name: str = "default"
) -> LogicalNode:
    """Convenience function to build a logical plan tree."""
    builder = LogicalPlanBuilder(catalog, db_name)
    return builder.build(stmt)
