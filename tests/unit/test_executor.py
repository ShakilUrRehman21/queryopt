"""
Unit tests for packages/executor (Physical operators and Volcano execution model).
"""
from pathlib import Path
from common.types import Column, DataType, ExecutionStatus, Schema, SortDirection
from executor.aggregate import HashAggregate
from executor.executor import VolcanoExecutor
from executor.filter import Filter
from executor.join import HashJoin, NestedLoopJoin
from executor.limit import Limit
from executor.project import Project
from executor.scan import SeqScan
from executor.sort import Sort
from parser.ast_types import BinaryOp, ColumnRef, FunctionCall, Literal, ProjectionItem, SortItem
from storage.table import Database


def test_volcano_scan_filter_project(tmp_path: Path):
    db = Database.create(tmp_path / "exec_db")
    schema = Schema([
        Column(name="id", data_type=DataType.INTEGER),
        Column(name="val", data_type=DataType.INTEGER),
    ])
    table = db.create_table("nums", schema)
    for i in range(1, 11):
        table.insert([i, i * 10])

    scan_op = SeqScan(table, schema)
    # Filter: val > 50
    pred = BinaryOp(left=ColumnRef("val"), op=">", right=Literal(50, DataType.INTEGER))
    filter_op = Filter(scan_op, pred)

    # Project: id
    out_schema = Schema([Column("id", DataType.INTEGER)])
    project_op = Project(filter_op, [ProjectionItem(expr=ColumnRef("id"))], out_schema)

    executor = VolcanoExecutor()
    result = executor.execute(project_op)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.rows_returned == 5
    assert result.rows == [(6,), (7,), (8,), (9,), (10,)]


def test_hash_join_and_nested_loop_join(tmp_path: Path):
    db = Database.create(tmp_path / "join_db")
    u_schema = Schema([Column("id", DataType.INTEGER), Column("name", DataType.TEXT)])
    o_schema = Schema([Column("order_id", DataType.INTEGER), Column("user_id", DataType.INTEGER)])

    u_tbl = db.create_table("users", u_schema)
    o_tbl = db.create_table("orders", o_schema)

    u_tbl.insert([1, "Alice"])
    u_tbl.insert([2, "Bob"])
    u_tbl.insert([3, "Charlie"])

    o_tbl.insert([101, 1])
    o_tbl.insert([102, 1])
    o_tbl.insert([103, 2])

    join_cond = BinaryOp(ColumnRef("id", "users"), "=", ColumnRef("user_id", "orders"))
    combined_schema = Schema(list(u_schema.columns) + list(o_schema.columns))

    # Test HashJoin
    hj = HashJoin(SeqScan(u_tbl, u_schema), SeqScan(o_tbl, o_schema), join_cond, combined_schema)
    res_hj = VolcanoExecutor().execute(hj)
    assert res_hj.rows_returned == 3

    # Test NestedLoopJoin
    nlj = NestedLoopJoin(SeqScan(u_tbl, u_schema), SeqScan(o_tbl, o_schema), join_cond, combined_schema)
    res_nlj = VolcanoExecutor().execute(nlj)
    assert res_nlj.rows_returned == 3
