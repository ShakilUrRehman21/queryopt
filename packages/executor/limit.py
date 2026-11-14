"""
executor.limit — Limit physical operator.

Skips an optional offset number of rows, then yields at most `limit` rows
before signaling EOF to downstream operators.
"""
from __future__ import annotations

from common.types import Row, Schema
from executor.base import PhysicalOperator


class Limit(PhysicalOperator):
    """
    Physical operator for LIMIT and OFFSET clauses.
    """

    def __init__(
        self,
        child: PhysicalOperator,
        limit: int,
        offset: int | None = None,
        estimated_rows: float = 0.0,
    ) -> None:
        super().__init__(schema=child.schema, estimated_rows=estimated_rows)
        self.child = child
        self.limit = limit
        self.offset = offset or 0
        self._count = 0

    @property
    def children(self) -> list[PhysicalOperator]:
        return [self.child]

    def open(self) -> None:
        super().open()
        self.child.open()
        self._count = 0

        # Skip offset rows
        for _ in range(self.offset):
            row = self.child.next()
            if row is None:
                break

    def next(self) -> Row | None:
        if self._count >= self.limit:
            return None

        row = self.child.next()
        if row is None:
            return None

        self._count += 1
        self.actual_rows += 1
        return row

    def close(self) -> None:
        self.child.close()
        super().close()
