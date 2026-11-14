"""
Unit tests for packages/planner (Logical Plan building and tree printing).
"""
from pathlib import Path
from common.types import Column, DataType, Schema
from parser.parser import parse
from planner.builder import build_logical_plan
from planner.nodes import LogicalFilter, LogicalLimit, LogicalProject, LogicalScan, LogicalSort
from planner.printer import explain_logical_plan
from storage.table import Database


def test_build_logical_plan_hierarchy(tmp_path: Path):
    db = Database.create(tmp_path / "plan_db")
    schema = Schema([
        Column(name="id", data_type=DataType.INTEGER),
        Column(name="score", data_type=DataType.FLOAT),
        Column(name="name", data_type=DataType.TEXT),
    ])
    db.create_table("students", schema)

    sql = "SELECT id, name FROM students WHERE score > 80.0 ORDER BY id DESC LIMIT 5"
    parsed = parse(sql)
    # pyrefly: ignore [bad-argument-type]
    plan = build_logical_plan(parsed.statement, db.catalog)

    # Top should be Limit
    assert isinstance(plan, LogicalLimit)
    assert plan.limit == 5

    # Child of Limit should be Sort
    assert isinstance(plan.child, LogicalSort)

    # Explain output should contain all nodes
    explained = explain_logical_plan(plan)
    assert "LogicalLimit" in explained
    assert "LogicalSort" in explained
    assert "LogicalFilter" in explained
    assert "LogicalScan" in explained
