"""
executor.scan — Sequential table scan operator (SeqScan).

Scans rows from a TableHandle one page at a time. Supports column pruning
to only project requested columns into the output Row.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from common.types import Row, Schema
from executor.base import PhysicalOperator

if TYPE_CHECKING:
    from storage.table import TableHandle


class SeqScan(PhysicalOperator):
    """
    Physical operator for sequential table scans.
    """

    def __init__(
        self,
        table_handle: TableHandle,
        schema: Schema,
        projection_columns: list[str] | None = None,
        estimated_rows: float = 0.0,
        alias: str | None = None,
    ) -> None:
        super().__init__(schema=schema, estimated_rows=estimated_rows)
        self.table_handle = table_handle
        self.table_name = table_handle.name
        self.alias = alias
        self.projection_columns = projection_columns
        self._iterator: Iterator[Row] | None = None

        # Build column index projection mapping if pruning is requested
        self._col_indices: list[int] | None = None
        if self.projection_columns:
            base_schema = table_handle.schema
            self._col_indices = [
                base_schema.column_index(col.split(".")[-1])
                for col in self.projection_columns
            ]

    @property
    def children(self) -> list[PhysicalOperator]:
        return []

    def open(self) -> None:
        super().open()
        self._iterator = self.table_handle.scan()

    def next(self) -> Row | None:
        if self._iterator is None:
            return None
        try:
            row = next(self._iterator)
            self.actual_rows += 1
            if self._col_indices is not None:
                return tuple(row[idx] for idx in self._col_indices)
            return row
        except StopIteration:
            return None

    def close(self) -> None:
        self._iterator = None
        super().close()
