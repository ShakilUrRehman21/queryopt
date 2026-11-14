"""
executor.join — Physical join operators (NestedLoopJoin, HashJoin).

Implements:
- NestedLoopJoin: General join operator supporting arbitrary conditions and INNER/LEFT join.
- HashJoin: Specialized equi-join operator using an in-memory hash table for the build side.
"""
from __future__ import annotations

from typing import Any

from common.errors import ExecutionError
from common.types import Column, DataType, JoinType, Row, Schema
from executor.base import PhysicalOperator, eval_expr
from parser.ast_types import BinaryOp, ColumnRef, Expr


def _get_op_tables(op: PhysicalOperator) -> set[str]:
    tables: set[str] = set()
    if hasattr(op, "table_name") and op.table_name:
        tables.add(op.table_name.lower())
    if hasattr(op, "alias") and op.alias:
        tables.add(op.alias.lower())
    for ch in getattr(op, "children", []):
        tables.update(_get_op_tables(ch))
    return tables


def _extract_equi_join_keys(
    condition: Expr,
    left_schema: Schema,
    right_schema: Schema,
    left_tables: set[str] | None = None,
    right_tables: set[str] | None = None,
) -> tuple[Expr, Expr] | None:
    """
    If condition is an equality between a left expression and right expression,
    return (left_expr, right_expr).
    """
    if isinstance(condition, BinaryOp) and condition.op == "=":
        # Check if one side references left and the other references right
        def get_side(expr: Expr) -> str | None:
            if isinstance(expr, ColumnRef):
                col_name = expr.column.lower()
                tbl_name = expr.table.lower() if expr.table else None
                if tbl_name and left_tables is not None and right_tables is not None:
                    if tbl_name in left_tables and tbl_name not in right_tables:
                        return "left"
                    if tbl_name in right_tables and tbl_name not in left_tables:
                        return "right"
                in_left = any(
                    (tbl_name and c.name.lower() == f"{tbl_name}.{col_name}") or
                    c.name.lower().split(".")[-1] == col_name
                    for c in left_schema.columns
                )
                in_right = any(
                    (tbl_name and c.name.lower() == f"{tbl_name}.{col_name}") or
                    c.name.lower().split(".")[-1] == col_name
                    for c in right_schema.columns
                )
                if in_left and not in_right:
                    return "left"
                if in_right and not in_left:
                    return "right"
            return None

        side_a = get_side(condition.left)
        side_b = get_side(condition.right)
        if side_a == "left" and side_b == "right":
            return condition.left, condition.right
        if side_a == "right" and side_b == "left":
            return condition.right, condition.left
    return None


class NestedLoopJoin(PhysicalOperator):
    """
    Nested loop join: for each outer row, iterates over materialized inner rows.
    """

    def __init__(
        self,
        left: PhysicalOperator,
        right: PhysicalOperator,
        condition: Expr,
        schema: Schema,
        join_type: JoinType = JoinType.INNER,
        estimated_rows: float = 0.0,
    ) -> None:
        super().__init__(schema=schema, estimated_rows=estimated_rows)
        self.left = left
        self.right = right
        self.condition = condition
        self.join_type = join_type

        self._inner_rows: list[Row] = []
        self._current_left_row: Row | None = None
        self._inner_index = 0
        self._left_matched = False

    @property
    def children(self) -> list[PhysicalOperator]:
        return [self.left, self.right]

    def open(self) -> None:
        super().open()
        self.left.open()
        self.right.open()

        # Materialize inner table into memory
        self._inner_rows = []
        while True:
            r = self.right.next()
            if r is None:
                break
            self._inner_rows.append(r)

        self._current_left_row = self.left.next()
        self._inner_index = 0
        self._left_matched = False

    def next(self) -> Row | None:
        while self._current_left_row is not None:
            # Check remaining inner rows for current left row
            while self._inner_index < len(self._inner_rows):
                right_row = self._inner_rows[self._inner_index]
                self._inner_index += 1

                combined_row = self._current_left_row + right_row
                if eval_expr(self.condition, combined_row, self.schema) is True:
                    self._left_matched = True
                    self.actual_rows += 1
                    return combined_row

            # Inner rows exhausted for this left row
            if self.join_type == JoinType.LEFT and not self._left_matched:
                null_right = tuple(None for _ in self.right.schema.columns)
                result = self._current_left_row + null_right
                self._advance_left()
                self.actual_rows += 1
                return result

            self._advance_left()

        return None

    def _advance_left(self) -> None:
        self._current_left_row = self.left.next()
        self._inner_index = 0
        self._left_matched = False

    def close(self) -> None:
        self._inner_rows.clear()
        self._current_left_row = None
        self.left.close()
        self.right.close()
        super().close()


class HashJoin(PhysicalOperator):
    """
    In-memory Hash Join for equi-join conditions.
    Builds a hash table on the right relation and streams the left relation.
    """

    def __init__(
        self,
        left: PhysicalOperator,
        right: PhysicalOperator,
        condition: Expr,
        schema: Schema,
        join_type: JoinType = JoinType.INNER,
        estimated_rows: float = 0.0,
    ) -> None:
        super().__init__(schema=schema, estimated_rows=estimated_rows)
        self.left = left
        self.right = right
        self.condition = condition
        self.join_type = join_type

        # Extract equi-join keys
        left_tables = _get_op_tables(left)
        right_tables = _get_op_tables(right)
        keys = _extract_equi_join_keys(condition, left.schema, right.schema, left_tables, right_tables)
        if keys is None:
            raise ExecutionError("HashJoin requires an equi-join condition (e.g. left.id = right.id)")
        self.left_key_expr, self.right_key_expr = keys

        self._hash_table: dict[Any, list[Row]] = {}
        self._current_left_row: Row | None = None
        self._matching_inner_rows: list[Row] = []
        self._match_idx = 0
        self._left_matched = False

    @property
    def children(self) -> list[PhysicalOperator]:
        return [self.left, self.right]

    def open(self) -> None:
        super().open()
        self.left.open()
        self.right.open()

        # Build phase: build hash table on right table
        self._hash_table.clear()
        while True:
            r = self.right.next()
            if r is None:
                break
            key = eval_expr(self.right_key_expr, r, self.right.schema)
            if key is not None:
                if key not in self._hash_table:
                    self._hash_table[key] = []
                self._hash_table[key].append(r)

        self._advance_left()

    def next(self) -> Row | None:
        while self._current_left_row is not None:
            while self._match_idx < len(self._matching_inner_rows):
                right_row = self._matching_inner_rows[self._match_idx]
                self._match_idx += 1
                combined_row = self._current_left_row + right_row
                if eval_expr(self.condition, combined_row, self.schema) is True:
                    self._left_matched = True
                    self.actual_rows += 1
                    return combined_row

            # If LEFT JOIN and no inner matches were found
            if self.join_type == JoinType.LEFT and not self._left_matched:
                null_right = tuple(None for _ in self.right.schema.columns)
                result = self._current_left_row + null_right
                self._advance_left()
                self.actual_rows += 1
                return result

            self._advance_left()

        return None

    def _advance_left(self) -> None:
        self._current_left_row = self.left.next()
        self._matching_inner_rows = []
        self._match_idx = 0
        self._left_matched = False

        if self._current_left_row is not None:
            key = eval_expr(self.left_key_expr, self._current_left_row, self.left.schema)
            if key is not None and key in self._hash_table:
                self._matching_inner_rows = self._hash_table[key]

    def close(self) -> None:
        self._hash_table.clear()
        self._matching_inner_rows = []
        self._current_left_row = None
        self.left.close()
        self.right.close()
        super().close()
