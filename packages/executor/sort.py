"""
executor.sort — Sort physical operator.

Materializes input rows in memory, sorts them according to ORDER BY sort items,
and streams the sorted rows to downstream operators.
"""
from __future__ import annotations

from typing import Any
from functools import cmp_to_key

from common.types import Row, Schema, SortDirection
from executor.base import PhysicalOperator, eval_expr
from parser.ast_types import SortItem


class Sort(PhysicalOperator):
    """
    Physical operator for sorting rows.
    """

    def __init__(
        self,
        child: PhysicalOperator,
        sort_items: list[SortItem],
        estimated_rows: float = 0.0,
    ) -> None:
        super().__init__(schema=child.schema, estimated_rows=estimated_rows)
        self.child = child
        self.sort_items = sort_items
        self._sorted_rows: list[Row] = []
        self._row_idx = 0

    @property
    def children(self) -> list[PhysicalOperator]:
        return [self.child]

    def open(self) -> None:
        super().open()
        self.child.open()

        # Materialize all child rows
        rows: list[Row] = []
        while True:
            row = self.child.next()
            if row is None:
                break
            rows.append(row)

        # Comparison function for sorting
        def compare_rows(a: Row, b: Row) -> int:
            for item in self.sort_items:
                val_a = eval_expr(item.expr, a, self.child.schema)
                val_b = eval_expr(item.expr, b, self.child.schema)

                # NULL handling (NULLs last)
                if val_a is None and val_b is None:
                    continue
                if val_a is None:
                    return 1 if item.direction == SortDirection.ASC else -1
                if val_b is None:
                    return -1 if item.direction == SortDirection.ASC else 1

                if val_a == val_b:
                    continue

                if val_a < val_b:
                    return -1 if item.direction == SortDirection.ASC else 1
                else:
                    return 1 if item.direction == SortDirection.ASC else -1
            return 0

        rows.sort(key=cmp_to_key(compare_rows))
        self._sorted_rows = rows
        self._row_idx = 0

    def next(self) -> Row | None:
        if self._row_idx < len(self._sorted_rows):
            row = self._sorted_rows[self._row_idx]
            self._row_idx += 1
            self.actual_rows += 1
            return row
        return None

    def close(self) -> None:
        self._sorted_rows.clear()
        self.child.close()
        super().close()
