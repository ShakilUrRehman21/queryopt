"""
Unit tests for packages/common (types, errors, config, logging).
"""
import pytest
from common.types import Column, DataType, Schema, make_row, row_to_dict
from common.errors import QueryOptError, TableNotFoundError, SQLParseError
from common.config import QueryOptConfig, Environment, ResourceLimits


def test_data_type_properties():
    assert DataType.INTEGER.is_numeric()
    assert DataType.FLOAT.is_numeric()
    assert not DataType.TEXT.is_numeric()
    assert DataType.INTEGER.is_comparable_to(DataType.FLOAT)
    assert DataType.NULL.is_comparable_to(DataType.TEXT)


def test_schema_and_column():
    col1 = Column(name="id", data_type=DataType.INTEGER, primary_key=True)
    col2 = Column(name="val", data_type=DataType.TEXT)
    schema = Schema([col1, col2])

    assert schema.column_names() == ["id", "val"]
    assert schema.column_index("val") == 1
    assert schema.has_column("id")
    assert not schema.has_column("nonexistent")

    with pytest.raises(ValueError):
        # duplicate column names
        Schema([col1, col1])


def test_make_row_and_dict():
    col1 = Column(name="id", data_type=DataType.INTEGER)
    col2 = Column(name="val", data_type=DataType.TEXT, default_value="default")
    schema = Schema([col1, col2])

    row = make_row(schema, {"id": 42})
    assert row == (42, "default")
    assert row_to_dict(schema, row) == {"id": 42, "val": "default"}


def test_error_hierarchy():
    err = TableNotFoundError(message="Table missing", table_name="users")
    assert isinstance(err, QueryOptError)
    assert err.error_code == "TABLE_NOT_FOUND"
    d = err.to_dict()
    assert d["error_code"] == "TABLE_NOT_FOUND"
    assert d["context"]["table_name"] == "users"
