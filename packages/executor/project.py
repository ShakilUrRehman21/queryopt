"""
executor.project — Project physical operator.

Evaluates projection expressions for each incoming row, creating a new
row conforming to the operator's output schema.
"""
from __future__ import annotations

from common.types import Row, Schema
from executor.base import PhysicalOperator, eval_expr
from parser.ast_types import ProjectionItem


class Project(PhysicalOperator):
    """
    Physical operator for projections and expression calculations.
    """

    def __init__(
        self,
        child: PhysicalOperator,
        projections: list[ProjectionItem],
        schema: Schema,
        estimated_rows: float = 0.0,
    ) -> None:
        super().__init__(schema=schema, estimated_rows=estimated_rows)
        self.child = child
        self.projections = projections

    @property
    def children(self) -> list[PhysicalOperator]:
        return [self.child]

    def open(self) -> None:
        super().open()
        self.child.open()

    def next(self) -> Row | None:
        row = self.child.next()
        if row is None:
            return None
        projected = tuple(
            eval_expr(p.expr, row, self.child.schema) for p in self.projections
        )
        self.actual_rows += 1
        return projected

    def close(self) -> None:
        self.child.close()
        super().close()
