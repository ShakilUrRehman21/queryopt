"""
storage.page — Page abstraction for QueryOpt Lab's storage layer.

A Page is the fundamental unit of storage. Each page holds a fixed number
of rows (records). Pages are the unit of reading and writing to/from disk.

Design notes:
- PAGE_SIZE_ROWS is deliberately small in Phase 1 for simplicity.
  A production system would use byte-addressed fixed-size pages.
- Free space tracking is simplified: a page is "full" when it has
  PAGE_SIZE_ROWS records.
- Pages are serialized to JSON for Phase 1. The page format is
  intentionally isolated behind PageStore so it can be replaced later.

ADR-004 (Volcano executor) notes: pages are the input to SeqScan operators.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from common.types import Row

# Rows per page. Deliberately small for Phase 1 to exercise page iteration.
# Can be tuned without changing the page API.
PAGE_SIZE_ROWS: int = 1000


@dataclass
class Page:
    """
    A single storage page containing up to PAGE_SIZE_ROWS rows.

    Attributes:
        page_id: Unique stable identifier for this page.
        table_name: Which table this page belongs to.
        records: The rows stored on this page.
    """
    page_id: str
    table_name: str
    records: list[Row] = field(default_factory=list)

    @classmethod
    def new(cls, table_name: str) -> "Page":
        """Create a new empty page for the given table."""
        return cls(page_id=str(uuid.uuid4()), table_name=table_name)

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def is_full(self) -> bool:
        return len(self.records) >= PAGE_SIZE_ROWS

    @property
    def free_slots(self) -> int:
        return max(0, PAGE_SIZE_ROWS - len(self.records))

    def append(self, row: Row) -> bool:
        """
        Append a row to this page.

        Returns:
            True if the row was appended; False if the page is full.
        """
        if self.is_full:
            return False
        self.records.append(row)
        return True

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "page_id": self.page_id,
            "table_name": self.table_name,
            "records": [list(r) for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Page":
        """Deserialize from a JSON-compatible dictionary."""
        return cls(
            page_id=data["page_id"],
            table_name=data["table_name"],
            records=[tuple(r) for r in data["records"]],
        )
