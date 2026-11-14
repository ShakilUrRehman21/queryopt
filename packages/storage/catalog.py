"""
storage.catalog — Database and table catalog (schema registry).

The Catalog is the authoritative source of truth about:
- What databases exist
- What tables exist in each database
- Each table's schema (column names, types, constraints)

The Catalog does NOT store data. Data lives in the PageStore.

Design notes:
- Catalog state is persisted as a single JSON file (catalog.json) in the
  database directory.
- Table names are case-insensitive (stored lowercase).
- The Catalog holds no locks; it is designed for single-process access
  in Phase 1. Concurrent access is a non-goal.
- Column order in Schema is the authoritative column order for all
  row tuples throughout the engine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from common.errors import (
    ColumnNotFoundError,
    StorageError,
    TableAlreadyExistsError,
    TableNotFoundError,
)
from common.logging import get_logger
from common.types import Column, DataType, Schema

logger = get_logger(__name__)

CATALOG_FILENAME = "catalog.json"


# ── Serialization helpers ─────────────────────────────────────────────────────

def _column_to_dict(col: Column) -> dict:
    return {
        "name": col.name,
        "data_type": col.data_type.value,
        "nullable": col.nullable,
        "primary_key": col.primary_key,
        "default_value": col.default_value,
    }


def _column_from_dict(d: dict) -> Column:
    return Column(
        name=d["name"],
        data_type=DataType(d["data_type"]),
        nullable=d.get("nullable", True),
        primary_key=d.get("primary_key", False),
        default_value=d.get("default_value"),
    )


# ── Table Metadata ────────────────────────────────────────────────────────────

@dataclass
class TableMeta:
    """
    Metadata for a single table.

    This is what the catalog knows about a table.
    It does NOT include the actual row data.
    """
    name: str
    schema: Schema
    page_ids: list[str] = field(default_factory=list)
    row_count: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "columns": [_column_to_dict(c) for c in self.schema.columns],
            "page_ids": self.page_ids,
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableMeta":
        columns = [_column_from_dict(c) for c in d["columns"]]
        return cls(
            name=d["name"],
            schema=Schema(columns),
            page_ids=d.get("page_ids", []),
            row_count=d.get("row_count", 0),
        )


# ── Catalog ───────────────────────────────────────────────────────────────────

class Catalog:
    """
    The schema registry for a single QueryOpt Lab database.

    A Catalog manages table metadata for one database directory.
    Multiple Catalog instances = multiple databases.

    Usage:
        catalog = Catalog.open("./data/research_db")
        catalog.create_table("orders", Schema([...]))
        meta = catalog.get_table("orders")
    """

    def __init__(self, db_dir: str | Path) -> None:
        self._db_dir = Path(db_dir)
        self._tables: dict[str, TableMeta] = {}
        self._catalog_path = self._db_dir / CATALOG_FILENAME

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, db_dir: str | Path) -> "Catalog":
        """
        Create a new empty database directory and catalog.
        Raises StorageError if the directory already contains a catalog.
        """
        db_path = Path(db_dir)
        catalog_path = db_path / CATALOG_FILENAME
        if catalog_path.exists():
            raise StorageError(
                message=f"Catalog already exists at {catalog_path}. Use Catalog.open() instead.",
                context={"path": str(catalog_path)},
            )
        db_path.mkdir(parents=True, exist_ok=True)
        cat = cls(db_dir)
        cat._persist()
        logger.info(f"Created new database at {db_path}")
        return cat

    @classmethod
    def open(cls, db_dir: str | Path) -> "Catalog":
        """
        Open an existing database catalog from disk.
        Raises StorageError if the catalog file does not exist.
        """
        db_path = Path(db_dir)
        catalog_path = db_path / CATALOG_FILENAME
        if not catalog_path.exists():
            raise StorageError(
                message=f"No catalog found at {catalog_path}. Use Catalog.create() for a new database.",
                context={"path": str(catalog_path)},
            )
        cat = cls(db_dir)
        cat._load()
        logger.debug(f"Opened database at {db_path} ({len(cat._tables)} tables)")
        return cat

    @classmethod
    def open_or_create(cls, db_dir: str | Path) -> "Catalog":
        """Open if exists, create if not. Useful for development and tests."""
        db_path = Path(db_dir)
        catalog_path = db_path / CATALOG_FILENAME
        if catalog_path.exists():
            return cls.open(db_dir)
        return cls.create(db_dir)

    # ── Table Management ─────────────────────────────────────────────────────

    def create_table(self, name: str, schema: Schema) -> TableMeta:
        """
        Register a new table in this catalog.

        Raises:
            TableAlreadyExistsError: if a table with this name already exists.
        """
        normalized = name.lower()
        if normalized in self._tables:
            raise TableAlreadyExistsError(
                message=f"Table '{name}' already exists.",
                table_name=name,
            )
        meta = TableMeta(name=normalized, schema=schema)
        self._tables[normalized] = meta
        self._persist()
        logger.info(f"Created table '{normalized}' with {len(schema)} columns")
        return meta

    def get_table(self, name: str) -> TableMeta:
        """
        Retrieve table metadata by name.

        Raises:
            TableNotFoundError: if no table with this name exists.
        """
        normalized = name.lower()
        if normalized not in self._tables:
            raise TableNotFoundError(
                message=f"Table '{name}' does not exist.",
                table_name=name,
            )
        return self._tables[normalized]

    def has_table(self, name: str) -> bool:
        return name.lower() in self._tables

    def list_tables(self) -> list[str]:
        return sorted(self._tables.keys())

    def drop_table(self, name: str) -> None:
        """
        Remove a table from the catalog.
        NOTE: This does NOT delete the underlying page files. Use PageStore for that.
        """
        normalized = name.lower()
        if normalized not in self._tables:
            raise TableNotFoundError(
                message=f"Table '{name}' does not exist.",
                table_name=name,
            )
        del self._tables[normalized]
        self._persist()
        logger.info(f"Dropped table '{normalized}' from catalog")

    def register_new_page(self, table_name: str, page_id: str) -> None:
        """Record that a new page has been allocated for a table."""
        meta = self.get_table(table_name)
        meta.page_ids.append(page_id)
        self._persist()

    def increment_row_count(self, table_name: str, count: int = 1) -> None:
        """Increment the tracked row count for a table after inserts."""
        meta = self.get_table(table_name)
        meta.row_count += count
        self._persist()

    def iter_tables(self) -> Iterator[TableMeta]:
        yield from self._tables.values()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """Write the current catalog state to disk."""
        data = {
            "version": "1",
            "tables": {name: meta.to_dict() for name, meta in self._tables.items()},
        }
        tmp_path = self._catalog_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(self._catalog_path)  # atomic on POSIX; best-effort on Windows
        except OSError as e:
            raise StorageError(
                message=f"Failed to write catalog to {self._catalog_path}: {e}",
                context={"path": str(self._catalog_path)},
            ) from e

    def _load(self) -> None:
        """Load catalog state from disk."""
        try:
            text = self._catalog_path.read_text(encoding="utf-8")
            data = json.loads(text)
        except OSError as e:
            raise StorageError(
                message=f"Failed to read catalog from {self._catalog_path}: {e}",
                context={"path": str(self._catalog_path)},
            ) from e
        except json.JSONDecodeError as e:
            raise StorageError(
                message=f"Catalog file is corrupt: {e}",
                context={"path": str(self._catalog_path)},
            ) from e

        self._tables = {}
        for name, table_data in data.get("tables", {}).items():
            self._tables[name] = TableMeta.from_dict(table_data)

    @property
    def db_dir(self) -> Path:
        return self._db_dir

    def __repr__(self) -> str:
        return f"Catalog(db_dir={self._db_dir}, tables={self.list_tables()})"
