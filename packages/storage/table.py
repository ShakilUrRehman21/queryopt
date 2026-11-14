"""
storage.table — High-level table operations (insert, scan, statistics).

TableHandle is the primary interface the rest of the engine uses to
interact with stored data. It combines Catalog metadata with PageStore
I/O and type coercion.

Design notes:
- TableHandle is not a database connection — it's a handle to a specific
  table within an already-opened database.
- Type coercion on insert is deliberate: the storage layer enforces
  that values match the declared column types.
- Sequential scan yields one Row at a time to support the Volcano
  iterator model without loading all rows into memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from storage.statistics import TableStatistics

from common.errors import ExecutionError, StorageError
from common.logging import get_logger
from common.types import DataType, Row, Schema, make_row
from storage.catalog import Catalog, TableMeta
from storage.page import Page
from storage.page_store import PageStore

logger = get_logger(__name__)


# ── Type Coercion ─────────────────────────────────────────────────────────────

def _coerce_value(value: Any, dtype: DataType, col_name: str) -> Any:
    """
    Coerce a Python value to the declared column type.

    Raises ExecutionError on type incompatibility.
    NULL (None) is always allowed for nullable columns; the nullable
    check happens in make_row().
    """
    if value is None:
        return None
    try:
        match dtype:
            case DataType.INTEGER:
                return int(value)
            case DataType.FLOAT:
                return float(value)
            case DataType.BOOLEAN:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            case DataType.TEXT:
                return str(value)
            case DataType.DATE | DataType.TIMESTAMP:
                return str(value)  # stored as ISO string; validation deferred
            case _:
                return value
    except (ValueError, TypeError) as e:
        raise ExecutionError(
            message=f"Cannot coerce value {value!r} to {dtype.value} for column '{col_name}': {e}",
            context={"column": col_name, "value": repr(value), "target_type": dtype.value},
        ) from e


# ── TableHandle ───────────────────────────────────────────────────────────────

class TableHandle:
    """
    Operations on a single table: insert, sequential scan, statistics collection.

    Obtain via Database.get_table(name).
    """

    def __init__(self, meta: TableMeta, page_store: PageStore, catalog: Catalog) -> None:
        self._meta = meta
        self._page_store = page_store
        self._catalog = catalog
        self._current_page: Page | None = None  # cached last page for appending

    @property
    def name(self) -> str:
        return self._meta.name

    @property
    def schema(self) -> Schema:
        return self._meta.schema

    @property
    def row_count(self) -> int:
        return self._meta.row_count

    @property
    def page_count(self) -> int:
        return len(self._meta.page_ids)

    # ── Insert ────────────────────────────────────────────────────────────────

    def insert(self, values: dict[str, Any] | list[Any]) -> None:
        """
        Insert a single row into the table.

        Args:
            values: Either a dict mapping column_name → value,
                    or a list of values in schema column order.
        """
        if isinstance(values, list):
            # Positional: zip with schema columns
            if len(values) != len(self.schema):
                raise ExecutionError(
                    message=(
                        f"INSERT into '{self.name}': expected {len(self.schema)} values, "
                        f"got {len(values)}."
                    ),
                )
            values_dict = {
                col.name: values[i]
                for i, col in enumerate(self.schema.columns)
            }
        else:
            values_dict = values

        # Coerce types
        coerced: dict[str, Any] = {}
        for col in self.schema.columns:
            raw = values_dict.get(col.name)
            coerced[col.name] = _coerce_value(raw, col.data_type, col.name)

        row = make_row(self.schema, coerced)
        self._append_row(row)

    def insert_many(self, rows: list[dict[str, Any] | list[Any]]) -> int:
        """Insert multiple rows. Returns the number of rows inserted."""
        for row_data in rows:
            self.insert(row_data)
        return len(rows)

    def _append_row(self, row: Row) -> None:
        """Append a row to the current tail page, allocating a new page if needed."""
        # Load or allocate the current tail page
        if self._current_page is None:
            if self._meta.page_ids:
                # Load last existing page to try appending
                last_id = self._meta.page_ids[-1]
                self._current_page = self._page_store.load_page(self.name, last_id)
            else:
                # No pages yet — allocate first
                self._current_page = self._page_store.new_page(self.name)
                self._catalog.register_new_page(self.name, self._current_page.page_id)

        if not self._current_page.append(row):
            # Current page is full — allocate a new one
            self._page_store.save_page(self._current_page)
            self._current_page = self._page_store.new_page(self.name)
            self._catalog.register_new_page(self.name, self._current_page.page_id)
            self._current_page.append(row)

        # Persist the updated page
        self._page_store.save_page(self._current_page)
        self._catalog.increment_row_count(self.name)

    # ── Sequential Scan ───────────────────────────────────────────────────────

    def scan(self) -> Iterator[Row]:
        """
        Yield all rows in the table in storage order (no guarantees on sort order).

        This is the primary input to SeqScan physical operators.
        Pages are loaded one at a time; the full table is never held in memory.
        """
        # Flush the in-memory tail page first
        if self._current_page is not None:
            self._page_store.save_page(self._current_page)

        for page in self._page_store.iter_pages(self.name, self._meta.page_ids):
            yield from page.records

    # ── Statistics ────────────────────────────────────────────────────────────

    def collect_statistics(self) -> "TableStatistics":
        """
        Compute basic column statistics by performing a full table scan.

        This is expensive (O(n) in row count) and should be called explicitly,
        not automatically on every query.
        """
        from storage.statistics import TableStatistics, ColumnStatsBuilder
        builders: dict[str, ColumnStatsBuilder] = {
            col.name: ColumnStatsBuilder(col) for col in self.schema.columns
        }
        row_count = 0
        for row in self.scan():
            row_count += 1
            for i, col in enumerate(self.schema.columns):
                builders[col.name].observe(row[i])

        return TableStatistics(
            table_name=self.name,
            row_count=row_count,
            column_stats={name: b.build() for name, b in builders.items()},
        )

    def __repr__(self) -> str:
        return f"TableHandle(name={self.name!r}, rows={self.row_count}, pages={self.page_count})"


# ── Database ──────────────────────────────────────────────────────────────────

class Database:
    """
    Top-level entry point for interacting with a QueryOpt Lab database.

    A Database holds a Catalog and a PageStore for one database directory.
    All SQL execution flows through a Database to access table data.

    Usage:
        db = Database.create("./data/research_db")
        db = Database.open("./data/research_db")
        db = Database.open_or_create("./data/research_db")
    """

    def __init__(self, catalog: Catalog, page_store: PageStore) -> None:
        self._catalog = catalog
        self._page_store = page_store
        self._table_handles: dict[str, TableHandle] = {}

    @classmethod
    def create(cls, db_dir: str | Path) -> "Database":
        """Create a new database. Raises StorageError if already exists."""
        catalog = Catalog.create(db_dir)
        page_store = PageStore(db_dir)
        return cls(catalog, page_store)

    @classmethod
    def open(cls, db_dir: str | Path) -> "Database":
        """Open an existing database. Raises StorageError if not found."""
        catalog = Catalog.open(db_dir)
        page_store = PageStore(db_dir)
        return cls(catalog, page_store)

    @classmethod
    def open_or_create(cls, db_dir: str | Path) -> "Database":
        """Open if exists, create if not. Useful for development."""
        catalog = Catalog.open_or_create(db_dir)
        page_store = PageStore(db_dir)
        return cls(catalog, page_store)

    def create_table(self, name: str, schema: Schema) -> TableHandle:
        """Create a table and return its handle."""
        meta = self._catalog.create_table(name, schema)
        handle = TableHandle(meta, self._page_store, self._catalog)
        self._table_handles[name.lower()] = handle
        return handle

    def get_table(self, name: str) -> TableHandle:
        """Get a handle to an existing table."""
        normalized = name.lower()
        if normalized not in self._table_handles:
            meta = self._catalog.get_table(normalized)
            self._table_handles[normalized] = TableHandle(meta, self._page_store, self._catalog)
        return self._table_handles[normalized]

    def has_table(self, name: str) -> bool:
        return self._catalog.has_table(name)

    def list_tables(self) -> list[str]:
        return self._catalog.list_tables()

    def drop_table(self, name: str) -> None:
        """Remove a table and delete its page files."""
        meta = self._catalog.get_table(name)
        self._page_store.delete_table_pages(name, meta.page_ids)
        self._catalog.drop_table(name)
        self._table_handles.pop(name.lower(), None)

    @property
    def catalog(self) -> Catalog:
        return self._catalog

    @property
    def db_dir(self) -> Path:
        return self._catalog.db_dir

    def __repr__(self) -> str:
        return f"Database(dir={self.db_dir}, tables={self.list_tables()})"
