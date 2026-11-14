"""
optimizer.annotator — Cardinality Annotator for logical plan trees.

Walks a LogicalNode tree bottom-up, attaches `estimated_rows` to each node
using the UniformCardinalityEstimator and table statistics from the catalog.

This is the bridge between the statistics package and the optimizer:
  1. Gather TableStatistics for every base table referenced in the plan.
  2. Traverse bottom-up, estimating each node's output row count.
  3. Return an annotated copy of the plan with all estimated_rows filled in.

Design notes:
- Creates new node copies via `with_children()` rather than mutating in place.
- Falls back to default estimates whenever stats are unavailable.
- Phase 2: Uses UniformCardinalityEstimator; Phase 3 will swap in
  HistogramCardinalityEstimator once equi-depth histograms are built.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from common.logging import get_logger
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
from query_stats.estimator import (
    CardinalityEstimator,
    HistogramCardinalityEstimator,
    UniformCardinalityEstimator,
)

if TYPE_CHECKING:
    from storage.catalog import Catalog
    from storage.statistics import TableStatistics

logger = get_logger(__name__)

# Default row estimate when no statistics exist at all.
_DEFAULT_ROW_ESTIMATE = 100.0


class CardinalityAnnotator:
    """
    Annotates each node in a LogicalNode tree with `estimated_rows`.

    The annotator gathers available TableStatistics from the catalog and
    uses a CardinalityEstimator (HistogramCardinalityEstimator by default)
    to estimate output cardinality for filter, join, and aggregate operators.
    """

    def __init__(
        self,
        catalog: Catalog,
        db_name: str = "default",
        estimator: CardinalityEstimator | None = None,
    ) -> None:
        self.catalog = catalog
        self.db_name = db_name
        self._estimator = estimator or HistogramCardinalityEstimator()
        # Statistics gathered by the most recent annotate() call; reused by the
        # CBO join enumerator so join estimates see the same column stats.
        self.table_stats: dict[str, TableStatistics] = {}

    @property
    def estimator(self) -> CardinalityEstimator:
        return self._estimator

    def annotate(self, root: LogicalNode) -> LogicalNode:
        """
        Traverse the logical plan tree bottom-up and return an annotated copy.

        Each node in the returned tree will have `estimated_rows` set.
        """
        table_stats = self._collect_table_stats(root)
        self.table_stats = table_stats
        annotated, _ = self._annotate_node(root, table_stats)
        return annotated

    # ── Private helpers ───────────────────────────────────────────────────────

    def _collect_table_stats(
        self, root: LogicalNode
    ) -> dict[str, TableStatistics]:
        """
        Walk the plan tree to find all LogicalScan nodes and load their stats.

        Returns a mapping: {table_name_or_alias -> TableStatistics}.
        """
        stats_map: dict[str, TableStatistics] = {}
        self._visit_scans(root, stats_map)
        return stats_map

    def _visit_scans(
        self, node: LogicalNode, stats_map: dict[str, TableStatistics]
    ) -> None:
        if isinstance(node, LogicalScan):
            try:
                from storage.statistics import TableStatistics
                t_stat = TableStatistics.load(
                    self.catalog.db_dir, node.table_name
                )
                if t_stat is not None:
                    # Register under physical name
                    stats_map[node.table_name.lower()] = t_stat
                    # Also register under alias so expressions can match
                    if node.alias:
                        stats_map[node.alias.lower()] = t_stat
            except Exception as exc:
                logger.debug(
                    f"Could not load stats for '{node.table_name}': {exc}"
                )
        for child in node.children:
            self._visit_scans(child, stats_map)

    def _annotate_node(
        self,
        node: LogicalNode,
        table_stats: dict[str, TableStatistics],
    ) -> tuple[LogicalNode, float]:
        """
        Recursively annotate a node and return (annotated_node, estimated_rows).
        """
        match node:
            case LogicalScan():
                row_estimate = self._scan_rows(node, table_stats)
                annotated = LogicalScan(
                    schema=node.schema,
                    table_name=node.table_name,
                    alias=node.alias,
                    projection_columns=list(node.projection_columns),
                    estimated_rows=row_estimate,
                )
                return annotated, row_estimate

            case LogicalFilter():
                ann_child, child_rows = self._annotate_node(node.child, table_stats)
                est = self._estimator.estimate_filter(
                    child_rows, node.predicate, table_stats
                )
                row_estimate = est.estimated_rows
                new_node = LogicalFilter(
                    schema=node.schema,
                    child=ann_child,
                    predicate=node.predicate,
                    estimated_rows=row_estimate,
                )
                return new_node, row_estimate

            case LogicalProject():
                ann_child, child_rows = self._annotate_node(node.child, table_stats)
                # Project does not change row count, only column set
                row_estimate = child_rows
                new_node = LogicalProject(
                    schema=node.schema,
                    child=ann_child,
                    projections=list(node.projections),
                    estimated_rows=row_estimate,
                )
                return new_node, row_estimate

            case LogicalJoin():
                ann_left, left_rows = self._annotate_node(node.left, table_stats)
                ann_right, right_rows = self._annotate_node(node.right, table_stats)
                est = self._estimator.estimate_join(
                    left_rows, right_rows, node.condition, table_stats
                )
                row_estimate = est.estimated_rows
                new_node = LogicalJoin(
                    schema=node.schema,
                    left=ann_left,
                    right=ann_right,
                    join_type=node.join_type,
                    condition=node.condition,
                    estimated_rows=row_estimate,
                )
                return new_node, row_estimate

            case LogicalAggregate():
                ann_child, child_rows = self._annotate_node(node.child, table_stats)
                est = self._estimator.estimate_aggregate(
                    child_rows, node.group_by, table_stats
                )
                row_estimate = est.estimated_rows
                new_node = LogicalAggregate(
                    schema=node.schema,
                    child=ann_child,
                    group_by=list(node.group_by),
                    aggregates=list(node.aggregates),
                    estimated_rows=row_estimate,
                )
                return new_node, row_estimate

            case LogicalSort():
                ann_child, child_rows = self._annotate_node(node.child, table_stats)
                # Sort preserves row count
                new_node = LogicalSort(
                    schema=node.schema,
                    child=ann_child,
                    sort_items=list(node.sort_items),
                    estimated_rows=child_rows,
                )
                return new_node, child_rows

            case LogicalLimit():
                ann_child, child_rows = self._annotate_node(node.child, table_stats)
                row_estimate = min(child_rows, float(node.limit))
                new_node = LogicalLimit(
                    schema=node.schema,
                    child=ann_child,
                    limit=node.limit,
                    offset=node.offset,
                    estimated_rows=row_estimate,
                )
                return new_node, row_estimate

            case _:
                # Generic fallback: annotate children, pass through count
                ann_children = []
                last_rows = _DEFAULT_ROW_ESTIMATE
                for c in node.children:
                    ann_c, r = self._annotate_node(c, table_stats)
                    ann_children.append(ann_c)
                    last_rows = r
                new_node = node.with_children(ann_children)
                return new_node, last_rows

    def _scan_rows(
        self,
        node: LogicalScan,
        table_stats: dict[str, TableStatistics],
    ) -> float:
        """Look up row count from stats, else fall back to a default."""
        key = node.table_name.lower()
        ts = table_stats.get(key)
        if ts is not None and ts.row_count > 0:
            return float(ts.row_count)
        return _DEFAULT_ROW_ESTIMATE
