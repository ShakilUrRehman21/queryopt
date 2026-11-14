"""
executor.base — Abstract PhysicalOperator and expression evaluation runtime.

Every physical operator implements the classical Volcano iterator model:
- `open()`: Prepares the operator, allocates buffers, opens child operators.
- `next()`: Returns the next `Row` (tuple of values) or `None` when EOF is reached.
- `close()`: Releases memory, files, and closes child operators.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from common.errors import ExecutionError
from common.types import DataType, OperatorStats, Row, Schema
from parser.ast_types import (
    BinaryOp,
    ColumnRef,
    Expr,
    FunctionCall,
    Literal,
    Star,
    UnaryOp,
)


def eval_expr(expr: Expr, row: Row | None, schema: Schema) -> Any:
    """
    Evaluate an expression against a single row using its Schema.
    """
    match expr:
        case Literal():
            return expr.value

        case ColumnRef():
            if row is None:
                raise ExecutionError(f"Cannot evaluate column '{expr}' against None row.")
            # Search by exact name or qualified name
            target = expr.column.lower()
            table_qualifier = expr.table.lower() if expr.table else None

            # First attempt: if table is qualified, check exact table.column
            if table_qualifier:
                for idx, col in enumerate(schema.columns):
                    if col.name.lower() == f"{table_qualifier}.{target}":
                        return row[idx]

            # Second attempt: check simple column name match
            for idx, col in enumerate(schema.columns):
                # col.name might be "t.col" or "col"
                col_name_parts = col.name.lower().split(".")
                if table_qualifier and len(col_name_parts) > 1 and col_name_parts[0] != table_qualifier:
                    continue
                if col_name_parts[-1] == target:
                    return row[idx]

            raise ExecutionError(
                f"Column '{expr}' not found in runtime row schema: {schema.column_names()}"
            )

        case Star():
            return row

        case BinaryOp():
            left_val = eval_expr(expr.left, row, schema)
            right_val = eval_expr(expr.right, row, schema)

            # NULL handling
            if left_val is None or right_val is None:
                if expr.op == "AND":
                    # In SQL: FALSE and NULL is FALSE
                    if left_val is False or right_val is False:
                        return False
                    return None
                if expr.op == "OR":
                    # In SQL: TRUE or NULL is TRUE
                    if left_val is True or right_val is True:
                        return True
                    return None
                return None

            op = expr.op.upper()
            if op == "=":
                return left_val == right_val
            elif op == "!=":
                return left_val != right_val
            elif op == "<":
                return left_val < right_val
            elif op == "<=":
                return left_val <= right_val
            elif op == ">":
                return left_val > right_val
            elif op == ">=":
                return left_val >= right_val
            elif op == "AND":
                return bool(left_val) and bool(right_val)
            elif op == "OR":
                return bool(left_val) or bool(right_val)
            elif op == "+":
                return left_val + right_val
            elif op == "-":
                return left_val - right_val
            elif op == "*":
                return left_val * right_val
            elif op == "/":
                if right_val == 0:
                    return None  # Or raise division by zero
                return left_val / right_val
            else:
                raise ExecutionError(f"Unsupported binary operator in execution: {expr.op}")

        case UnaryOp():
            val = eval_expr(expr.expr, row, schema)
            op = expr.op.upper()
            if op == "IS NULL":
                return val is None
            elif op == "IS NOT NULL":
                return val is not None
            elif op == "NOT":
                return None if val is None else (not bool(val))
            elif op == "-":
                return None if val is None else (-val)
            else:
                raise ExecutionError(f"Unsupported unary operator in execution: {expr.op}")

        case FunctionCall():
            fname = expr.name.upper()
            if row is not None and fname in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
                for idx, col in enumerate(schema.columns):
                    c_name = col.name.upper()
                    if c_name == fname or c_name == str(expr).upper() or c_name == expr.name.upper():
                        return row[idx]

            # Scalar math/string functions
            args = [eval_expr(a, row, schema) for a in expr.args]
            if fname == "ABS":
                return abs(args[0]) if args[0] is not None else None
            elif fname == "LOWER":
                return str(args[0]).lower() if args[0] is not None else None
            elif fname == "UPPER":
                return str(args[0]).upper() if args[0] is not None else None
            elif fname == "LENGTH":
                return len(str(args[0])) if args[0] is not None else None
            else:
                raise ExecutionError(f"Unsupported scalar function call: {fname}")

        case _:
            raise ExecutionError(f"Cannot evaluate unsupported expression: {type(expr).__name__}")


class PhysicalOperator(ABC):
    """
    Abstract base class for all Volcano physical execution operators.
    """

    def __init__(self, schema: Schema, estimated_rows: float = 0.0) -> None:
        self.schema = schema
        self.estimated_rows = estimated_rows
        self.operator_id = f"{self.__class__.__name__}_{uuid4().hex[:8]}"
        self.actual_rows = 0
        self.execution_time_ms = 0.0
        self._start_time: float | None = None
        self._is_open = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def open(self) -> None:
        """Initialize the operator and its children."""
        self._is_open = True
        self.actual_rows = 0
        self._start_time = time.perf_counter()

    @abstractmethod
    def next(self) -> Row | None:
        """Fetch the next row or None if exhausted."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up operator state and close children."""
        if self._start_time is not None:
            self.execution_time_ms += (time.perf_counter() - self._start_time) * 1000.0
            self._start_time = None
        self._is_open = False

    @property
    @abstractmethod
    def children(self) -> list[PhysicalOperator]:
        """Return all child physical operators."""
        ...

    def get_stats(self) -> OperatorStats:
        """Produce runtime statistics for this operator."""
        return OperatorStats(
            operator_id=self.operator_id,
            operator_type=self.__class__.__name__,
            estimated_rows=self.estimated_rows,
            actual_rows=self.actual_rows,
            execution_time_ms=self.execution_time_ms,
        )

    def collect_all_stats(self) -> dict[str, OperatorStats]:
        """Recursively collect stats from this operator and all descendants."""
        result = {self.operator_id: self.get_stats()}
        for child in self.children:
            result.update(child.collect_all_stats())
        return result
