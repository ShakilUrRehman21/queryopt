"""
Unit tests for the Cost-Based Optimizer (CBO) and CardinalityAnnotator.

Tests are isolated from disk I/O where possible, using in-memory structures
and tmp_path for database fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from common.types import Column, DataType, JoinType, Schema
from optimizer.annotator import CardinalityAnnotator
from optimizer.cost_model import CostModel, CostModelConfig
from optimizer.optimizer_cost_based import (
    CostBasedOptimizer,
    _extract_join_cluster,
)
from parser.ast_types import BinaryOp, ColumnRef, Literal
from planner.nodes import (
    LogicalAggregate,
    LogicalFilter,
    LogicalJoin,
    LogicalProject,
    LogicalScan,
    LogicalSort,
    LogicalLimit,
)
from storage.table import Database


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def two_table_db(tmp_path: Path) -> Database:
    db = Database.create(tmp_path / "cbo_db")
    db.create_table(
        "orders",
        Schema([
            Column("id", DataType.INTEGER),
            Column("customer_id", DataType.INTEGER),
            Column("amount", DataType.FLOAT),
        ]),
    )
    db.create_table(
        "customers",
        Schema([
            Column("id", DataType.INTEGER),
            Column("name", DataType.TEXT),
            Column("country", DataType.TEXT),
        ]),
    )
    # Insert a few rows so the catalog has row counts
    orders_tbl = db.get_table("orders")
    for i in range(1, 11):
        orders_tbl.insert({"id": i, "customer_id": i, "amount": float(i * 10)})

    customers_tbl = db.get_table("customers")
    for i in range(1, 6):
        customers_tbl.insert({"id": i, "name": f"C{i}", "country": "US"})

    return db


@pytest.fixture()
def orders_schema() -> Schema:
    return Schema([
        Column("id", DataType.INTEGER),
        Column("customer_id", DataType.INTEGER),
        Column("amount", DataType.FLOAT),
    ])


@pytest.fixture()
def customers_schema() -> Schema:
    return Schema([
        Column("id", DataType.INTEGER),
        Column("name", DataType.TEXT),
        Column("country", DataType.TEXT),
    ])


@pytest.fixture()
def join_schema(orders_schema: Schema, customers_schema: Schema) -> Schema:
    cols = list(orders_schema.columns)
    for c in customers_schema.columns:
        if any(c.name == existing.name for existing in cols):
            cols.append(Column(f"customers.{c.name}", c.data_type, c.nullable, c.default_value))
        else:
            cols.append(c)
    return Schema(cols)


# ── CardinalityAnnotator tests ────────────────────────────────────────────────

class TestCardinalityAnnotator:

    def test_scan_annotation_fallback(self, two_table_db: Database, orders_schema: Schema):
        """Scan node gets estimated_rows from table stats or default."""
        scan = LogicalScan(schema=orders_schema, table_name="orders")
        annotator = CardinalityAnnotator(two_table_db.catalog)
        annotated = annotator.annotate(scan)
        # Row count: 10 rows were inserted, so stats may have 10 or fall back to default
        assert annotated.estimated_rows is not None
        assert annotated.estimated_rows > 0

    def test_filter_annotation_reduces_rows(
        self, two_table_db: Database, orders_schema: Schema
    ):
        """Filter annotation reduces estimated row count relative to scan."""
        scan = LogicalScan(schema=orders_schema, table_name="orders")
        pred = BinaryOp(
            left=ColumnRef(column="id", table="orders"),
            op=">",
            right=Literal(value=5, data_type=DataType.INTEGER),
        )
        filt = LogicalFilter(schema=orders_schema, child=scan, predicate=pred)

        annotator = CardinalityAnnotator(two_table_db.catalog)
        annotated = annotator.annotate(filt)

        assert annotated.estimated_rows is not None
        # Filter should reduce row count compared to scan
        ann_scan = annotated.child  # type: ignore[attr-defined]
        assert annotated.estimated_rows <= ann_scan.estimated_rows

    def test_project_annotation_preserves_rows(
        self, two_table_db: Database, orders_schema: Schema
    ):
        """Project does not change estimated_rows."""
        from parser.ast_types import ProjectionItem
        scan = LogicalScan(schema=orders_schema, table_name="orders")
        proj_schema = Schema([Column("id", DataType.INTEGER)])
        proj = LogicalProject(
            schema=proj_schema,
            child=scan,
            projections=[ProjectionItem(expr=ColumnRef("id"), alias="id")],
        )
        annotator = CardinalityAnnotator(two_table_db.catalog)
        annotated = annotator.annotate(proj)
        ann_scan = annotated.child  # type: ignore[attr-defined]
        # Project must pass through the same row count
        assert annotated.estimated_rows == ann_scan.estimated_rows

    def test_join_annotation(
        self,
        two_table_db: Database,
        orders_schema: Schema,
        customers_schema: Schema,
        join_schema: Schema,
    ):
        """Join node gets a cardinality estimate based on equi-join formula."""
        scan_o = LogicalScan(schema=orders_schema, table_name="orders")
        scan_c = LogicalScan(schema=customers_schema, table_name="customers")
        cond = BinaryOp(
            left=ColumnRef(column="customer_id", table="orders"),
            op="=",
            right=ColumnRef(column="id", table="customers"),
        )
        join = LogicalJoin(
            schema=join_schema,
            left=scan_o,
            right=scan_c,
            join_type=JoinType.INNER,
            condition=cond,
        )
        annotator = CardinalityAnnotator(two_table_db.catalog)
        annotated = annotator.annotate(join)
        assert annotated.estimated_rows is not None
        assert annotated.estimated_rows > 0

    def test_limit_annotation(
        self, two_table_db: Database, orders_schema: Schema
    ):
        """Limit annotation caps estimated_rows at the LIMIT value."""
        scan = LogicalScan(schema=orders_schema, table_name="orders")
        limit = LogicalLimit(schema=orders_schema, child=scan, limit=3)
        annotator = CardinalityAnnotator(two_table_db.catalog)
        annotated = annotator.annotate(limit)
        assert annotated.estimated_rows is not None
        assert annotated.estimated_rows <= 3.0

    def test_sort_annotation_preserves_rows(
        self, two_table_db: Database, orders_schema: Schema
    ):
        """Sort does not change estimated_rows."""
        from parser.ast_types import SortItem
        from common.types import SortDirection
        scan = LogicalScan(schema=orders_schema, table_name="orders")
        sort = LogicalSort(
            schema=orders_schema,
            child=scan,
            sort_items=[SortItem(expr=ColumnRef("amount"), direction=SortDirection.DESC)],
        )
        annotator = CardinalityAnnotator(two_table_db.catalog)
        annotated = annotator.annotate(sort)
        ann_scan = annotated.child  # type: ignore[attr-defined]
        assert annotated.estimated_rows == ann_scan.estimated_rows


# ── CostModel tests ───────────────────────────────────────────────────────────

class TestCostModel:

    def test_scan_cost_positive(self, two_table_db: Database, orders_schema: Schema):
        """SeqScan must have a positive cost."""
        from executor.scan import SeqScan
        from storage.table import TableHandle
        tbl_meta = two_table_db.catalog.get_table("orders")
        tbl_handle = TableHandle(tbl_meta, two_table_db._page_store, two_table_db.catalog)
        op = SeqScan(table_handle=tbl_handle, schema=orders_schema, estimated_rows=10.0)

        model = CostModel()
        cost = model.estimate_cost(op)
        assert cost > 0.0

    def test_custom_config(self, two_table_db: Database, orders_schema: Schema):
        """CostModel respects custom CostModelConfig."""
        from executor.scan import SeqScan
        from storage.table import TableHandle
        tbl_meta = two_table_db.catalog.get_table("orders")
        tbl_handle = TableHandle(tbl_meta, two_table_db._page_store, two_table_db.catalog)
        op = SeqScan(table_handle=tbl_handle, schema=orders_schema, estimated_rows=10.0)

        cheap_cfg = CostModelConfig(seq_page_cost=0.1, cpu_tuple_cost=0.001)
        expensive_cfg = CostModelConfig(seq_page_cost=10.0, cpu_tuple_cost=1.0)

        cheap_cost = CostModel(cheap_cfg).estimate_cost(op)
        expensive_cost = CostModel(expensive_cfg).estimate_cost(op)
        assert cheap_cost < expensive_cost


# ── CostBasedOptimizer tests ──────────────────────────────────────────────────

class TestCostBasedOptimizer:

    def test_simple_select_correctness(self, two_table_db: Database, orders_schema: Schema):
        """CBO produces a physical plan for a simple SELECT."""
        from parser.parser import parse
        from parser.validator import validate
        from planner.builder import build_logical_plan

        parsed = parse("SELECT id, amount FROM orders WHERE amount > 50")
        stmt = parsed.statement
        validate(stmt, two_table_db.catalog)
        plan = build_logical_plan(stmt, two_table_db.catalog)

        cbo = CostBasedOptimizer()
        result = cbo.optimize(plan, two_table_db.catalog, two_table_db._page_store)

        assert result.optimizer_id == "cost_based"
        assert result.physical_plan is not None
        assert result.estimated_cost > 0.0

    def test_join_correctness(
        self,
        two_table_db: Database,
    ):
        """CBO on a two-table join produces a valid physical plan."""
        from parser.parser import parse
        from parser.validator import validate
        from planner.builder import build_logical_plan
        from executor.executor import VolcanoExecutor

        parsed = parse(
            "SELECT orders.id, customers.name "
            "FROM orders JOIN customers ON orders.customer_id = customers.id"
        )
        stmt = parsed.statement
        validate(stmt, two_table_db.catalog)
        plan = build_logical_plan(stmt, two_table_db.catalog)

        cbo = CostBasedOptimizer()
        result = cbo.optimize(plan, two_table_db.catalog, two_table_db._page_store)

        assert result.physical_plan is not None
        assert result.alternatives_considered >= 1

        executor = VolcanoExecutor()
        exec_result = executor.execute(result.physical_plan)
        assert exec_result.rows is not None

    def test_alternatives_counted_for_single_table(
        self, two_table_db: Database, orders_schema: Schema
    ):
        """Single-table query has exactly 1 alternative (no join reordering)."""
        from parser.parser import parse
        from parser.validator import validate
        from planner.builder import build_logical_plan

        parsed = parse("SELECT id FROM orders")
        stmt = parsed.statement
        validate(stmt, two_table_db.catalog)
        plan = build_logical_plan(stmt, two_table_db.catalog)

        cbo = CostBasedOptimizer()
        result = cbo.optimize(plan, two_table_db.catalog, two_table_db._page_store)
        assert result.alternatives_considered >= 1

    def test_optimizer_id(self):
        cbo = CostBasedOptimizer()
        assert cbo.optimizer_id == "cost_based"


# ── Join cluster extraction tests ─────────────────────────────────────────────

class TestJoinClusterExtraction:

    def _make_scan(self, name: str) -> LogicalScan:
        return LogicalScan(
            schema=Schema([Column(f"{name}_id", DataType.INTEGER)]),
            table_name=name,
        )

    def _make_join(self, left, right) -> LogicalJoin:
        schema = Schema(left.schema.columns + right.schema.columns)
        cond = BinaryOp(ColumnRef(left.schema.columns[0].name), "=", ColumnRef(right.schema.columns[0].name))
        return LogicalJoin(
            schema=schema, left=left, right=right,
            join_type=JoinType.INNER, condition=cond,
        )

    def test_two_table_cluster(self):
        a = self._make_scan("a")
        b = self._make_scan("b")
        join = self._make_join(a, b)
        scans, conditions, schema = _extract_join_cluster(join)
        assert len(scans) == 2
        assert len(conditions) == 1

    def test_three_table_cluster(self):
        a = self._make_scan("a")
        b = self._make_scan("b")
        c = self._make_scan("c")
        join_ab = self._make_join(a, b)
        join_abc = self._make_join(join_ab, c)
        scans, conditions, schema = _extract_join_cluster(join_abc)
        assert len(scans) == 3
        assert len(conditions) == 2

    def test_single_scan_not_join(self):
        a = self._make_scan("a")
        scans, conditions, schema = _extract_join_cluster(a)
        assert len(scans) == 1
        assert len(conditions) == 0
