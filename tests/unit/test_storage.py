"""
Unit tests for packages/storage (Database, Catalog, PageStore, TableHandle, Statistics).
"""
import pytest
from pathlib import Path
from common.types import Column, DataType, Schema
from storage.table import Database


def test_database_lifecycle_and_table_crud(tmp_path: Path):
    db_dir = tmp_path / "test_db"
    db = Database.create(db_dir)

    schema = Schema([
        Column(name="id", data_type=DataType.INTEGER, primary_key=True),
        Column(name="score", data_type=DataType.FLOAT),
        Column(name="name", data_type=DataType.TEXT),
    ])

    table = db.create_table("students", schema)
    assert db.has_table("students")
    assert db.list_tables() == ["students"]

    # Insert rows
    table.insert([1, 95.5, "Alice"])
    table.insert({"id": 2, "score": 88.0, "name": "Bob"})
    table.insert([3, 72.3, "Charlie"])

    assert table.row_count == 3

    # Scan rows
    rows = list(table.scan())
    assert len(rows) == 3
    assert rows[0] == (1, 95.5, "Alice")
    assert rows[1] == (2, 88.0, "Bob")
    assert rows[2] == (3, 72.3, "Charlie")

    # Statistics collection
    stats = table.collect_statistics()
    assert stats.row_count == 3
    assert "score" in stats.columns
    assert stats.columns["id"].distinct_count == 3
    assert stats.columns["score"].min_value == 72.3
    assert stats.columns["score"].max_value == 95.5

    # Re-open database
    db2 = Database.open(db_dir)
    assert db2.has_table("students")
    table2 = db2.get_table("students")
    assert table2.row_count == 3
    rows2 = list(table2.scan())
    assert len(rows2) == 3
