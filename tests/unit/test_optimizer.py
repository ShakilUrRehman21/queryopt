"""
Unit tests for packages/optimizer (Optimization rules and optimizers).
"""
from pathlib import Path
from common.types import Column, DataType, Schema
from optimizer.optimizer_naive import NaiveOptimizer
from optimizer.optimizer_rule_based import RuleBasedOptimizer
from optimizer.rules.constant_folding import ConstantFoldingRule
from parser.ast_types import BinaryOp, ColumnRef, Literal
from parser.parser import parse
from planner.builder import build_logical_plan
from planner.nodes import LogicalFilter, LogicalScan
from storage.table import Database


def test_constant_folding():
    # age > 10 + 5 -> age > 15
    pred = BinaryOp(
        left=ColumnRef("age"),
        op=">",
        right=BinaryOp(Literal(10, DataType.INTEGER), "+", Literal(5, DataType.INTEGER)),
    )
    schema = Schema([Column("age", DataType.INTEGER)])
    scan = LogicalScan(schema=schema, table_name="users")
    filt = LogicalFilter(schema=schema, child=scan, predicate=pred)

    rule = ConstantFoldingRule()
    opt_plan, modified = rule.apply(filt)
    assert modified
    assert isinstance(opt_plan, LogicalFilter)
    assert isinstance(opt_plan.predicate, BinaryOp)
    assert isinstance(opt_plan.predicate.right, Literal)
    assert opt_plan.predicate.right.value == 15


def test_rule_based_optimizer_execution(tmp_path: Path):
    db = Database.create(tmp_path / "opt_db")
    schema = Schema([Column("id", DataType.INTEGER), Column("val", DataType.TEXT)])
    db.create_table("t", schema)

    parsed = parse("SELECT id FROM t WHERE id = 1 + 1")
    # pyrefly: ignore [bad-argument-type]
    plan = build_logical_plan(parsed.statement, db.catalog)

    rbo = RuleBasedOptimizer()
    res = rbo.optimize(plan, db.catalog, db._page_store)

    assert "ConstantFolding" in res.rules_applied
    assert res.physical_plan is not None
