"""
Section 44 Optimization Correctness Oracle:
Verifies that any query optimized by RuleBasedOptimizer or CostBasedOptimizer
produces the exact same semantic result set as the Naive reference optimizer.
"""
from pathlib import Path
import pytest
from common.types import ExecutionStatus
from queryopt.engine import QueryEngine


@pytest.fixture()
def engine(tmp_path: Path) -> QueryEngine:
    eng = QueryEngine(data_dir=tmp_path, db_name="oracle_db")
    # Base tables
    eng.execute("CREATE TABLE t1 (id INTEGER, val INTEGER, tag TEXT)")
    eng.execute("CREATE TABLE t2 (id INTEGER, t1_id INTEGER, amount FLOAT)")
    # Populate
    for i in range(1, 21):
        eng.execute(f"INSERT INTO t1 VALUES ({i}, {i * 5}, 'tag_{i % 3}')")
        eng.execute(f"INSERT INTO t2 VALUES ({100 + i}, {i}, {float(i * 10)})")
    return eng


QUERIES = [
    "SELECT id, val FROM t1 WHERE val > 50",
    "SELECT id, tag FROM t1 WHERE id = 2 + 3",
    "SELECT t1.id, t2.amount FROM t1 JOIN t2 ON t1.id = t2.t1_id WHERE t1.val > 30",
    "SELECT tag, COUNT(id) FROM t1 GROUP BY tag ORDER BY tag ASC",
    "SELECT id, val FROM t1 ORDER BY val DESC LIMIT 5",
]


@pytest.mark.parametrize("query", QUERIES)
def test_rule_based_matches_naive(engine: QueryEngine, query: str):
    """Rule-based optimizer must produce the same result set as naive."""
    res_naive = engine.execute(query, optimizer_id="naive")
    res_rule = engine.execute(query, optimizer_id="rule_based")

    assert res_naive.status == ExecutionStatus.SUCCESS, f"Naive failed: {query}"
    assert res_rule.status == ExecutionStatus.SUCCESS, f"Rule-based failed: {query}"
    assert res_naive.rows == res_rule.rows, (
        f"Correctness violation [rule_based] on:\n{query}\n"
        f"Naive: {res_naive.rows}\nRule: {res_rule.rows}"
    )


@pytest.mark.parametrize("query", QUERIES)
def test_cost_based_matches_naive(engine: QueryEngine, query: str):
    """Cost-based optimizer must produce the same result set as naive."""
    res_naive = engine.execute(query, optimizer_id="naive")
    res_cbo = engine.execute(query, optimizer_id="cost_based")

    assert res_naive.status == ExecutionStatus.SUCCESS, f"Naive failed: {query}"
    assert res_cbo.status == ExecutionStatus.SUCCESS, f"CBO failed: {query}\n{res_cbo.error_message}"
    assert res_naive.rows == res_cbo.rows, (
        f"Correctness violation [cost_based] on:\n{query}\n"
        f"Naive: {res_naive.rows}\nCBO:   {res_cbo.rows}"
    )


def test_cost_based_produces_estimated_cost(engine: QueryEngine):
    """CBO OptimizationResult must expose a positive estimated_cost."""
    from parser.parser import parse
    from parser.ast_types import SelectStatement
    from parser.validator import validate
    from planner.builder import build_logical_plan
    from optimizer.optimizer_cost_based import CostBasedOptimizer

    sql = "SELECT t1.id, t2.amount FROM t1 JOIN t2 ON t1.id = t2.t1_id"
    parse_result = parse(sql)
    stmt = parse_result.statement
    assert isinstance(stmt, SelectStatement)

    validate(stmt, engine.db.catalog, engine.db_name)
    logical_plan = build_logical_plan(stmt, engine.db.catalog, engine.db_name)

    cbo = CostBasedOptimizer()
    result = cbo.optimize(
        logical_plan,
        catalog=engine.db.catalog,
        page_store=engine.db._page_store,
        db_name=engine.db_name,
    )

    assert result.estimated_cost > 0.0, "CBO must produce a positive estimated cost"
    assert result.optimizer_id == "cost_based"
    assert result.alternatives_considered >= 1


def test_cbo_cost_le_naive_for_join(engine: QueryEngine):
    """
    CBO plan cost should be <= naive plan cost for a join query,
    since CBO can choose a better join order while naive uses whatever
    the parser gave it.

    NOTE: With only two tables the join order is trivially the same,
    so this asserts cost is at worst equal (not strictly better).
    """
    from parser.parser import parse
    from parser.ast_types import SelectStatement
    from parser.validator import validate
    from planner.builder import build_logical_plan
    from optimizer.optimizer_cost_based import CostBasedOptimizer
    from optimizer.optimizer_naive import NaiveOptimizer
    from optimizer.cost_model import CostModel

    sql = "SELECT t1.id, t2.amount FROM t1 JOIN t2 ON t1.id = t2.t1_id"
    parse_result = parse(sql)
    stmt = parse_result.statement
    assert isinstance(stmt, SelectStatement)

    validate(stmt, engine.db.catalog, engine.db_name)
    logical_plan = build_logical_plan(stmt, engine.db.catalog, engine.db_name)

    from optimizer.annotator import CardinalityAnnotator

    cbo = CostBasedOptimizer()
    naive = NaiveOptimizer()
    cost_model = CostModel()

    annotator = CardinalityAnnotator(engine.db.catalog, engine.db_name)
    annotated_plan = annotator.annotate(logical_plan)

    cbo_result = cbo.optimize(
        logical_plan, engine.db.catalog, engine.db._page_store, engine.db_name
    )
    naive_result = naive.optimize(
        annotated_plan, engine.db.catalog, engine.db._page_store, engine.db_name
    )

    cbo_cost = cost_model.estimate_cost(cbo_result.physical_plan)
    naive_cost = cost_model.estimate_cost(naive_result.physical_plan)

    assert cbo_cost <= naive_cost * 1.01, (  # 1% tolerance for floating-point
        f"CBO cost ({cbo_cost:.4f}) should be <= naive cost ({naive_cost:.4f})"
    )
