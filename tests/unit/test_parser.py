"""
Unit tests for packages/parser (SQL parsing and AST validation).
"""
import pytest
from pathlib import Path
from common.errors import SQLParseError, SQLValidationError, TableNotFoundError
from common.types import Column, DataType, Schema
from parser.parser import parse
from parser.validator import validate
from parser.ast_types import SelectStatement, CreateTableStatement, InsertStatement
from storage.table import Database


def test_parse_create_table():
    res = parse("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL, age INTEGER)")
    assert isinstance(res.statement, CreateTableStatement)
    assert res.statement.table_name == "users"
    assert len(res.statement.columns) == 3
    assert res.statement.columns[0].primary_key is True


def test_parse_insert():
    res = parse("INSERT INTO users (id, email) VALUES (1, 'alice@example.com'), (2, 'bob@example.com')")
    assert isinstance(res.statement, InsertStatement)
    assert res.statement.table_name == "users"
    assert res.statement.columns == ["id", "email"]
    assert len(res.statement.values_rows) == 2


def test_parse_complex_select():
    sql = """
    SELECT u.id, u.email, COUNT(o.id) AS order_cnt
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE u.age >= 18 AND o.status = 'COMPLETED'
    GROUP BY u.id, u.email
    HAVING COUNT(o.id) > 5
    ORDER BY order_cnt DESC
    LIMIT 10 OFFSET 20
    """
    res = parse(sql)
    assert isinstance(res.statement, SelectStatement)
    stmt = res.statement
    assert stmt.from_table.table_name == "users"
    assert stmt.from_table.alias == "u"
    assert len(stmt.joins) == 1
    assert stmt.where is not None
    assert len(stmt.group_by) == 2
    assert stmt.having is not None
    assert len(stmt.order_by) == 1
    assert stmt.limit == 10
    assert stmt.offset == 20


def test_validator_detects_missing_table_and_column(tmp_path: Path):
    db = Database.create(tmp_path / "val_db")
    schema = Schema([Column(name="id", data_type=DataType.INTEGER)])
    db.create_table("items", schema)

    # Missing table
    res1 = parse("SELECT * FROM non_existent")
    with pytest.raises(TableNotFoundError):
        validate(res1.statement, db.catalog)

    # Missing column
    res2 = parse("SELECT bad_col FROM items")
    with pytest.raises(SQLValidationError):
        validate(res2.statement, db.catalog)
