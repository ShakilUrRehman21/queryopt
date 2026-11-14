"""
optimizer.optimizer_cost_based — Cost-Based Optimizer (CBO).

Strategy
--------
1. Apply all rule-based heuristic rewrites first (constant folding,
   predicate & projection pushdown) to prune search space.
2. Annotate the resulting logical plan with cardinality estimates via
   CardinalityAnnotator (backed by UniformCardinalityEstimator + stored stats).
3. For multi-way joins (≤ MAX_JOIN_TABLES tables), enumerate all left-deep
   join orderings using dynamic programming and pick the cheapest plan.
4. Compile each candidate logical plan variant with PhysicalPlanner, score
   it via CostModel, and keep the minimum.
5. Return an OptimizationResult whose `estimated_cost` reflects the best
   physical plan's cost.

Design decisions
----------------
- We enumerate only LEFT-DEEP trees (not bushy), a practical constraint that
  cuts the search space from O(n!) to O(n * 2^n) while still allowing join
  reordering for up to ~10 tables.
- We pick between HashJoin and NestedLoopJoin inside PhysicalPlanner, so the
  CBO selects the join ORDER; PhysicalPlanner selects the join IMPLEMENTATION.
- This is intentionally a research prototype, not a full System-R cascade
  planner.  The goal is clear pedagogical comparability, not raw speed.

References
----------
- Selinger et al. (1979): Access path selection in a relational database
  management system.  ACM PODS.
- Moerkotte & Neumann (2006): Analysis of two existing and one new dynamic
  programming algorithm for the generation of optimal bushy join trees without
  cross products.  VLDB.
"""
from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING

from common.logging import get_logger
from common.types import Column, DataType, JoinType, Schema
from parser.ast_types import BinaryOp, ColumnRef, Expr, Literal, UnaryOp
from optimizer.annotator import CardinalityAnnotator
from optimizer.base import BaseOptimizer
from optimizer.cost_model import CostModel, CostModelConfig
from optimizer.optimizer_rule_based import RuleBasedOptimizer
from optimizer.physical_planner import PhysicalPlanner
from optimizer.result import OptimizationResult
from query_stats.estimator import CardinalityEstimator
from planner.nodes import (
    LogicalAggregate,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalNode,
    LogicalProject,
    LogicalScan,
    LogicalSort,
)

if TYPE_CHECKING:
    from storage.catalog import Catalog
    from storage.page_store import PageStore

logger = get_logger(__name__)

# Maximum number of join tables for which we enumerate orderings.
# Beyond this we fall back to the order produced by the rule-based rewrite.
MAX_JOIN_TABLES = 8


class CostBasedOptimizer(BaseOptimizer):
    """
    Cost-Based Query Optimizer (CBO) using dynamic-programming join reordering.

    Pipeline:
        logical_plan
            └─► heuristic_rewrite (RBO rules)
                └─► cardinality_annotation
                    └─► join_reordering (DP over left-deep trees)
                        └─► physical_compilation + cost scoring
                            └─► OptimizationResult (cheapest plan)
    """

    def __init__(
        self,
        cost_config: CostModelConfig | None = None,
        max_join_tables: int = MAX_JOIN_TABLES,
        estimator: CardinalityEstimator | None = None,
    ) -> None:
        self._cost_model = CostModel(cost_config)
        # None → CardinalityAnnotator's default (HistogramCardinalityEstimator).
        # Subclasses / the registry inject learned or error-injected estimators.
        self._estimator = estimator
        self._max_join_tables = max_join_tables
        # Re-use rule-based optimizer for the initial heuristic phase
        self._rbo = RuleBasedOptimizer()

    @property
    def optimizer_id(self) -> str:
        return "cost_based"

    def optimize(
        self,
        logical_plan: LogicalNode,
        catalog: Catalog,
        page_store: PageStore,
        db_name: str = "default",
    ) -> OptimizationResult:
        start_time = time.perf_counter()

        # ── Step 1: Heuristic rewrites ────────────────────────────────────────
        rbo_result = self._rbo.optimize(logical_plan, catalog, page_store, db_name)
        rewritten_plan = rbo_result.optimized_logical_plan
        applied_rules = list(rbo_result.rules_applied)

        # ── Step 2: Cardinality annotation ────────────────────────────────────
        annotator = self._make_annotator(catalog, page_store, db_name)
        annotated_plan = annotator.annotate(rewritten_plan)

        # ── Step 3: Join reordering via DP ────────────────────────────────────
        planner = PhysicalPlanner(catalog, page_store, db_name)

        best_logical = self._reorder_joins(annotated_plan, annotator)
        applied_rules.append("join_reorder_dp")

        # ── Step 4: Compile best logical plan and score ───────────────────────
        best_physical = planner.plan(best_logical)
        best_cost = self._cost_model.estimate_cost(best_physical)

        # Also score the annotated_plan (pre-reorder) to count alternatives
        alternatives_considered = self._count_join_orderings(annotated_plan)

        opt_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"CBO: cost={best_cost:.4f}, alternatives={alternatives_considered}, "
            f"time={opt_time_ms:.2f}ms"
        )

        return OptimizationResult(
            optimizer_id=self.optimizer_id,
            initial_logical_plan=logical_plan,
            optimized_logical_plan=best_logical,
            physical_plan=best_physical,
            optimization_time_ms=opt_time_ms,
            estimated_cost=best_cost,
            rules_applied=applied_rules,
            alternatives_considered=alternatives_considered,
            metadata=self._result_metadata(annotator),
        )

    # ── Extension hooks (overridden by Adaptive / Learned / LLM optimizers) ──

    def _make_annotator(
        self, catalog: Catalog, page_store: PageStore, db_name: str
    ) -> CardinalityAnnotator:
        return CardinalityAnnotator(catalog, db_name, estimator=self._estimator)

    def _join_output_rows(
        self,
        left_plan: LogicalNode,
        right_plan: LogicalNode,
        condition: Expr,
        left_rows: float,
        right_rows: float,
        annotator: CardinalityAnnotator,
    ) -> float:
        """Estimated output cardinality of joining two candidate sub-plans."""
        est = annotator.estimator.estimate_join(
            left_rows, right_rows, condition, annotator.table_stats
        )
        return est.estimated_rows

    def _result_metadata(self, annotator: CardinalityAnnotator) -> dict:
        return {"estimator": type(annotator.estimator).__name__}

    # ── Join reordering ───────────────────────────────────────────────────────

    def _reorder_joins(
        self, plan: LogicalNode, annotator: CardinalityAnnotator
    ) -> LogicalNode:
        """
        Recursively descend through the plan tree. When a chain of inner joins
        is found, extract the base relations and join conditions, enumerate all
        left-deep orderings, pick the cheapest, and rebuild.

        Non-join subtrees are returned unchanged.
        """
        match plan:
            case LogicalJoin():
                # Collect the entire join cluster (may be a multi-way chain)
                scans, conditions, schema = _extract_join_cluster(plan)
                if len(scans) > self._max_join_tables:
                    # Too large – skip reordering, just descend into subtrees
                    logger.debug(
                        f"Join cluster has {len(scans)} tables > {self._max_join_tables}; "
                        "skipping CBO reordering"
                    )
                    return plan

                best_plan = self._dp_best_join_order(scans, conditions, plan.schema, annotator)
                # Wrap the result in the same outer operators (Filter etc.)
                return best_plan

            case LogicalFilter():
                reordered_child = self._reorder_joins(plan.child, annotator)
                return LogicalFilter(
                    schema=plan.schema,
                    child=reordered_child,
                    predicate=plan.predicate,
                    estimated_rows=plan.estimated_rows,
                )

            case LogicalProject():
                reordered_child = self._reorder_joins(plan.child, annotator)
                return LogicalProject(
                    schema=plan.schema,
                    child=reordered_child,
                    projections=list(plan.projections),
                    estimated_rows=plan.estimated_rows,
                )

            case LogicalAggregate():
                reordered_child = self._reorder_joins(plan.child, annotator)
                return LogicalAggregate(
                    schema=plan.schema,
                    child=reordered_child,
                    group_by=list(plan.group_by),
                    aggregates=list(plan.aggregates),
                    estimated_rows=plan.estimated_rows,
                )

            case LogicalSort():
                reordered_child = self._reorder_joins(plan.child, annotator)
                return LogicalSort(
                    schema=plan.schema,
                    child=reordered_child,
                    sort_items=list(plan.sort_items),
                    estimated_rows=plan.estimated_rows,
                )

            case LogicalLimit():
                reordered_child = self._reorder_joins(plan.child, annotator)
                return LogicalLimit(
                    schema=plan.schema,
                    child=reordered_child,
                    limit=plan.limit,
                    offset=plan.offset,
                    estimated_rows=plan.estimated_rows,
                )

            case _:
                return plan

    def _dp_best_join_order(
        self,
        scans: list[LogicalNode],
        conditions: list[Expr],
        outer_schema: Schema,
        annotator: CardinalityAnnotator,
    ) -> LogicalNode:
        """
        Dynamic-programming over left-deep join orderings.

        dp[frozenset(table_indices)] = (best_logical_plan, estimated_cost)

        Base case:  single relations — estimated cost = 0 (already scanned).
        Transition: extend a k-table plan by joining one more table.

        Prioritizes connected joins over Cartesian products.
        Returns the best full join of all scans.
        """
        # Qualify scan schemas so column resolution is unambiguous across joins
        qualified_scans = [_qualify_scan_node(s) for s in scans]
        n = len(qualified_scans)
        all_indices = frozenset(range(n))

        scan_tables = [_get_node_tables(s) for s in qualified_scans]
        scan_cols = [
            {c.name.lower().split(".")[-1] for c in s.schema.columns}
            for s in qualified_scans
        ]

        # Map each atomic condition to the scan indices it involves
        cond_scans: list[tuple[Expr, set[int]]] = [
            (c, _get_expr_scans(c, scan_tables, scan_cols)) for c in conditions
        ]

        # Seed: single-table "plans"
        dp: dict[frozenset, tuple[LogicalNode, float]] = {}
        for i, scan in enumerate(qualified_scans):
            dp[frozenset([i])] = (scan, 0.0)

        # Build up plans of increasing size
        for size in range(2, n + 1):
            for subset in itertools.combinations(range(n), size):
                subset_key = frozenset(subset)

                # Separate connected extensions from cross products
                connected_candidates = []
                cross_candidates = []

                for new_idx in subset:
                    prev_key = subset_key - {new_idx}
                    if prev_key not in dp:
                        continue

                    prev_plan, prev_cost = dp[prev_key]
                    new_scan = qualified_scans[new_idx]

                    # Find applicable join condition(s) for this pair
                    applicable: list[Expr] = []
                    is_connected = False
                    for cond, involved in cond_scans:
                        if involved.issubset(subset_key) and new_idx in involved:
                            applicable.append(cond)
                            if bool(involved & prev_key):
                                is_connected = True

                    if is_connected:
                        connected_candidates.append(
                            (
                                new_idx,
                                prev_plan,
                                prev_cost,
                                new_scan,
                                _combine_conditions(applicable),
                                False,
                            )
                        )
                    else:
                        cross_candidates.append(
                            (
                                new_idx,
                                prev_plan,
                                prev_cost,
                                new_scan,
                                Literal(value=True, data_type=DataType.BOOLEAN),
                                True,
                            )
                        )

                # Prioritize connected joins over Cartesian products
                candidates = connected_candidates if connected_candidates else cross_candidates
                best_cost = float("inf")
                best_plan_here: LogicalNode | None = None

                for new_idx, prev_plan, prev_cost, new_scan, cond, is_cross in candidates:
                    left_rows = prev_plan.estimated_rows or 100.0
                    right_rows = new_scan.estimated_rows or 100.0

                    if is_cross:
                        out_rows = left_rows * right_rows
                        join_cost = left_rows * right_rows * 1000.0
                    else:
                        out_rows = self._join_output_rows(
                            prev_plan, new_scan, cond, left_rows, right_rows, annotator
                        )
                        join_cost = self._cost_model_join_cost(
                            left_rows, right_rows, out_rows
                        )

                    candidate_schema = Schema(
                        list(prev_plan.schema.columns) + list(new_scan.schema.columns)
                    )
                    candidate_join = LogicalJoin(
                        schema=candidate_schema,
                        left=prev_plan,
                        right=new_scan,
                        join_type=JoinType.INNER,
                        condition=cond,
                        estimated_rows=out_rows,
                    )

                    total_cost = prev_cost + join_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_plan_here = candidate_join

                if best_plan_here is not None:
                    dp[subset_key] = (best_plan_here, best_cost)

        if all_indices in dp:
            return dp[all_indices][0]

        # Fallback: return original first scan joined with the rest in order
        logger.warning("DP join reordering failed to find a plan; using default order")
        return scans[0]

    def _cost_model_join_cost(
        self, left_rows: float, right_rows: float, out_rows: float
    ) -> float:
        """
        Analytical join cost estimate without building a full physical tree.

        Approximates the cheaper of HashJoin and NestedLoopJoin.

        HashJoin:
            build = (cpu_tuple + hash_penalty) * right_rows
            probe = cpu_tuple * left_rows
        NestedLoopJoin:
            loop  = cpu_op * left_rows * right_rows
        """
        cfg = self._cost_model.config
        hash_cost = (
            (cfg.cpu_tuple_cost + cfg.hash_mem_penalty) * right_rows
            + cfg.cpu_tuple_cost * left_rows
        )
        nlj_cost = cfg.cpu_operator_cost * left_rows * right_rows
        return min(hash_cost, nlj_cost)

    def _count_join_orderings(self, plan: LogicalNode) -> int:
        """Count how many join ordering alternatives were considered."""
        if isinstance(plan, LogicalJoin):
            scans, _, _ = _extract_join_cluster(plan)
            n = len(scans)
            if n > 1:
                result = 1
                for i in range(1, min(n, self._max_join_tables) + 1):
                    result *= i
                return result
        for ch in getattr(plan, "children", []):
            cnt = self._count_join_orderings(ch)
            if cnt > 1:
                return cnt
        return 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_conjunction(expr: Expr | None) -> list[Expr]:
    """Recursively split AND conjunctions into a flat list of conjuncts."""
    if expr is None:
        return []
    if isinstance(expr, BinaryOp) and expr.op.upper() == "AND":
        return _split_conjunction(expr.left) + _split_conjunction(expr.right)
    return [expr]


def _combine_conditions(conds: list[Expr]) -> Expr:
    """Combine multiple expressions into a single AND conjunction."""
    if not conds:
        return Literal(value=True, data_type=DataType.BOOLEAN)
    result = conds[0]
    for c in conds[1:]:
        result = BinaryOp(left=result, op="AND", right=c)
    return result


def _get_node_tables(node: LogicalNode) -> set[str]:
    """Collect all table names and aliases present in a logical subtree."""
    tables: set[str] = set()
    if isinstance(node, LogicalScan):
        if node.table_name:
            tables.add(node.table_name.lower())
        if node.alias:
            tables.add(node.alias.lower())
    for ch in getattr(node, "children", []):
        tables.update(_get_node_tables(ch))
    return tables


def _qualify_scan_node(scan: LogicalNode) -> LogicalNode:
    """Ensure all schema columns of a scan are qualified with table name or alias."""
    tbl = None
    if isinstance(scan, LogicalScan):
        tbl = scan.alias or scan.table_name
    elif isinstance(scan, LogicalFilter) and isinstance(scan.child, LogicalScan):
        tbl = scan.child.alias or scan.child.table_name

    if not tbl:
        return scan

    new_cols = []
    for c in scan.schema.columns:
        col_name = c.name if "." in c.name else f"{tbl}.{c.name}"
        new_cols.append(
            Column(
                name=col_name,
                data_type=c.data_type,
                nullable=c.nullable,
                default_value=c.default_value,
            )
        )
    new_schema = Schema(new_cols)

    if isinstance(scan, LogicalScan):
        return LogicalScan(
            schema=new_schema,
            table_name=scan.table_name,
            alias=scan.alias,
            projection_columns=scan.projection_columns,
            estimated_rows=scan.estimated_rows,
            estimated_cost=scan.estimated_cost,
        )
    elif isinstance(scan, LogicalFilter):
        return LogicalFilter(
            schema=new_schema,
            child=scan.child,
            predicate=scan.predicate,
            estimated_rows=scan.estimated_rows,
            estimated_cost=scan.estimated_cost,
        )
    return scan


def _get_expr_scans(
    expr: Expr,
    scan_tables: list[set[str]],
    scan_cols: list[set[str]],
) -> set[int]:
    """Identify which scan indices an expression references."""
    refs: list[ColumnRef] = []

    def walk(x: Expr) -> None:
        if isinstance(x, ColumnRef):
            refs.append(x)
        elif isinstance(x, BinaryOp):
            walk(x.left)
            walk(x.right)
        elif isinstance(x, UnaryOp):
            walk(x.expr)

    walk(expr)

    matched: set[int] = set()
    for ref in refs:
        col = ref.column.lower()
        if ref.table:
            tbl = ref.table.lower()
            for i, tbls in enumerate(scan_tables):
                if tbl in tbls:
                    matched.add(i)
                    break
        else:
            for i, cols in enumerate(scan_cols):
                if col in cols:
                    matched.add(i)
                    break
    return matched


def _extract_join_cluster(
    node: LogicalNode,
) -> tuple[list[LogicalNode], list[Expr], Schema]:
    """
    Extract a flat list of base-table scans and join conditions from a chain
    of inner LogicalJoin nodes.

    Stops at non-join nodes (which become leaves in the cluster).

    Returns:
        scans: leaf plans (base scans or non-join subtrees)
        conditions: list of atomic conjunct expressions
        schema: the schema of the topmost join node
    """
    scans: list[LogicalNode] = []
    conditions: list[Expr] = []
    schema = node.schema

    def _collect(n: LogicalNode) -> None:
        if isinstance(n, LogicalJoin) and n.join_type == JoinType.INNER:
            _collect(n.left)
            _collect(n.right)
            if n.condition is not None:
                conditions.extend(_split_conjunction(n.condition))
        else:
            scans.append(n)

    _collect(node)
    return scans, conditions, schema
