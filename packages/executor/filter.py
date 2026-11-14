"""
executor.filter — Filter physical operator.

Evaluates a boolean predicate on each row from its child operator and
yields only rows where the predicate evaluates to True (in accordance with SQL
three-valued logic).
"""
from __future__ import annotations

from common.types import Row, Schema
from executor.base import PhysicalOperator, eval_expr
from parser.ast_types import Expr


class Filter(PhysicalOperator):
    """
    Physical operator for filtering rows based on a predicate.
    """

    def __init__(
        self,
        child: PhysicalOperator,
        predicate: Expr,
        estimated_rows: float = 0.0,
    ) -> None:
        super().__init__(schema=child.schema, estimated_rows=estimated_rows)
        self.child = child
        self.predicate = predicate

    @property
    def children(self) -> list[PhysicalOperator]:
        return [self.child]

    def open(self) -> None:
        super().open()
        self.child.open()

    def next(self) -> Row | None:
        while True:
            row = self.child.next()
            if row is None:
                return None
            result = eval_expr(self.predicate, row, self.child.schema)
            if result is True:
                self.actual_rows += 1
                return row

    def close(self) -> None:
        self.child.close()
        super().close()
