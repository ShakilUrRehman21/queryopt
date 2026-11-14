"""
executor.aggregate — HashAggregate physical operator.

Computes aggregations (COUNT, SUM, AVG, MIN, MAX) over input rows,
optionally grouped by one or more GROUP BY expressions using an in-memory hash table.
"""
from __future__ import annotations

from typing import Any

from common.errors import ExecutionError
from common.types import Row, Schema
from executor.base import PhysicalOperator, eval_expr
from parser.ast_types import Expr, FunctionCall, ProjectionItem, Star


class _AggregateAccumulator:
    """Accumulates values for a single aggregate function instance."""

    def __init__(self, func_call: FunctionCall) -> None:
        self.func_name = func_call.name.upper()
        self.arg_expr = func_call.args[0] if func_call.args else None
        self.distinct = func_call.distinct
        self.is_star = isinstance(self.arg_expr, Star) or (self.func_name == "COUNT" and not func_call.args)

        self.count = 0
        self.total = 0.0
        self.min_val: Any = None
        self.max_val: Any = None
        self.seen_values: set[Any] = set()

    def update(self, val: Any) -> None:
        if self.is_star:
            self.count += 1
            return

        if val is None:
            return  # Aggregates ignore NULLs in SQL

        if self.distinct:
            if val in self.seen_values:
                return
            self.seen_values.add(val)

        self.count += 1

        if self.func_name in ("SUM", "AVG"):
            try:
                num = float(val)
                self.total += num
            except (ValueError, TypeError):
                pass

        if self.func_name == "MIN":
            if self.min_val is None or val < self.min_val:
                self.min_val = val

        if self.func_name == "MAX":
            if self.max_val is None or val > self.max_val:
                self.max_val = val

    def finalize(self) -> Any:
        match self.func_name:
            case "COUNT":
                return self.count
            case "SUM":
                return self.total if self.count > 0 else None
            case "AVG":
                return (self.total / self.count) if self.count > 0 else None
            case "MIN":
                return self.min_val
            case "MAX":
                return self.max_val
            case _:
                raise ExecutionError(f"Unknown aggregate function '{self.func_name}'")


class HashAggregate(PhysicalOperator):
    """
    Hash-based aggregation operator.
    """

    def __init__(
        self,
        child: PhysicalOperator,
        group_by: list[Expr],
        aggregates: list[ProjectionItem],
        schema: Schema,
        estimated_rows: float = 0.0,
    ) -> None:
        super().__init__(schema=schema, estimated_rows=estimated_rows)
        self.child = child
        self.group_by = group_by
        self.aggregates = aggregates

        # Extract function calls from aggregate items
        self._agg_calls: list[FunctionCall] = []
        for item in self.aggregates:
            if isinstance(item.expr, FunctionCall):
                self._agg_calls.append(item.expr)
            else:
                raise ExecutionError(f"Expected FunctionCall in aggregate item, got {item.expr}")

        self._result_rows: list[Row] = []
        self._row_idx = 0

    @property
    def children(self) -> list[PhysicalOperator]:
        return [self.child]

    def open(self) -> None:
        super().open()
        self.child.open()

        # Group key -> list of _AggregateAccumulator
        groups: dict[tuple[Any, ...], list[_AggregateAccumulator]] = {}
        has_rows = False

        while True:
            row = self.child.next()
            if row is None:
                break
            has_rows = True

            # Compute group key
            if self.group_by:
                key = tuple(eval_expr(expr, row, self.child.schema) for expr in self.group_by)
            else:
                key = ()

            if key not in groups:
                groups[key] = [_AggregateAccumulator(call) for call in self._agg_calls]

            # Update accumulators
            accs = groups[key]
            for idx, call in enumerate(self._agg_calls):
                val = None
                if call.args and not isinstance(call.args[0], Star):
                    val = eval_expr(call.args[0], row, self.child.schema)
                accs[idx].update(val)

        # Build output rows
        self._result_rows = []
        if not groups and not self.group_by:
            # Aggregate without GROUP BY over empty table emits 1 row of default values
            default_accs = [_AggregateAccumulator(call) for call in self._agg_calls]
            self._result_rows.append(tuple(acc.finalize() for acc in default_accs))
        else:
            for key, accs in groups.items():
                agg_results = tuple(acc.finalize() for acc in accs)
                # Output row is (group_keys..., agg_values...)
                out_row = key + agg_results
                self._result_rows.append(out_row)

        self._row_idx = 0

    def next(self) -> Row | None:
        if self._row_idx < len(self._result_rows):
            row = self._result_rows[self._row_idx]
            self._row_idx += 1
            self.actual_rows += 1
            return row
        return None

    def close(self) -> None:
        self._result_rows.clear()
        self.child.close()
        super().close()
