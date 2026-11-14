"""
optimizer.optimizer_adaptive — Between-query Adaptive Optimizer (Phase 6, RQ4).

Identical to the Cost-Based Optimizer except for where cardinalities come from:

    observed rows (feedback store)  ──►  if this exact sub-expression ran before
    modelled estimate (histograms)  ──►  otherwise

Because signatures are order-independent, a single execution teaches the
optimizer about *every* sub-expression it executed, and the DP join enumerator
can reuse those facts for candidate orders it has never run. This is Option A
from the Phase 0 plan (OQ5): statistics improve between queries; there is no
mid-query re-planning.

The returned `OptimizationResult.metadata` lists every estimate the feedback
replaced, so the UI and experiments can show *why* an adaptive plan differs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optimizer.annotator import CardinalityAnnotator
from optimizer.feedback import CardinalityFeedbackStore
from optimizer.optimizer_cost_based import CostBasedOptimizer
from optimizer.signature import logical_signature
from planner.nodes import LogicalFilter, LogicalJoin, LogicalNode

if TYPE_CHECKING:
    from parser.ast_types import Expr
    from storage.catalog import Catalog
    from storage.page_store import PageStore
    from storage.statistics import TableStatistics


class FeedbackAwareAnnotator(CardinalityAnnotator):
    """CardinalityAnnotator that overrides estimates with observed cardinalities."""

    def __init__(self, catalog: Catalog, db_name: str, store: CardinalityFeedbackStore, **kw: Any) -> None:
        super().__init__(catalog, db_name, **kw)
        self.store = store
        self.corrections: dict[str, dict[str, Any]] = {}

    def correct(self, node: LogicalNode, modelled: float) -> float:
        sig = logical_signature(node)
        entry = self.store.lookup(sig)
        if entry is None or sig is None:
            return modelled
        actual = max(float(entry.actual_rows), 1.0)
        self.corrections[sig.key] = {
            "tables": sorted(sig.tables),
            "predicates": sorted(sig.predicates),
            "modelled_rows": round(modelled, 2),
            "observed_rows": entry.actual_rows,
            "observations": entry.observations,
        }
        return actual

    def _annotate_node(
        self, node: LogicalNode, table_stats: dict[str, TableStatistics]
    ) -> tuple[LogicalNode, float]:
        new_node, rows = super()._annotate_node(node, table_stats)
        # Base scans are already exact (catalog row counts), so only filters
        # and joins are corrected.
        if isinstance(new_node, (LogicalFilter, LogicalJoin)):
            corrected = self.correct(new_node, rows)
            if corrected != rows:
                new_node.estimated_rows = corrected
                rows = corrected
        return new_node, rows


class AdaptiveOptimizer(CostBasedOptimizer):
    """Cost-based DP optimizer whose cardinalities are corrected by past executions."""

    @property
    def optimizer_id(self) -> str:
        return "adaptive"

    def _make_annotator(
        self, catalog: Catalog, page_store: PageStore, db_name: str
    ) -> CardinalityAnnotator:
        store = CardinalityFeedbackStore(catalog.db_dir)
        return FeedbackAwareAnnotator(catalog, db_name, store, estimator=self._estimator)

    def _join_output_rows(
        self,
        left_plan: LogicalNode,
        right_plan: LogicalNode,
        condition: Expr,
        left_rows: float,
        right_rows: float,
        annotator: CardinalityAnnotator,
    ) -> float:
        modelled = super()._join_output_rows(
            left_plan, right_plan, condition, left_rows, right_rows, annotator
        )
        if not isinstance(annotator, FeedbackAwareAnnotator):
            return modelled
        from common.types import JoinType

        probe = LogicalJoin(
            schema=left_plan.schema,
            left=left_plan,
            right=right_plan,
            join_type=JoinType.INNER,
            condition=condition,
        )
        return annotator.correct(probe, modelled)

    def _result_metadata(self, annotator: CardinalityAnnotator) -> dict:
        meta = super()._result_metadata(annotator)
        if isinstance(annotator, FeedbackAwareAnnotator):
            meta["feedback_entries"] = len(annotator.store)
            meta["feedback_corrections"] = list(annotator.corrections.values())
        return meta
