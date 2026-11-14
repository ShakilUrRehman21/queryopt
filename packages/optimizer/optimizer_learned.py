"""
optimizer.optimizer_learned — Cost-based optimizer driven by learned cardinalities.

Same DP join enumeration and cost model as `CostBasedOptimizer`; the only
difference is the estimator: `LearnedCardinalityEstimator`, which uses a
self-supervised per-table MLP (see `query_stats.learned`). Comparing this
optimizer with `cost_based` on the same engine isolates the effect of the
estimator — exactly the controlled experiment Persona 1 (Alex) asked for.

Models are trained lazily the first time a table is seen and cached on disk
next to the catalog (`learned_models/`). A model is retrained automatically if
the table's row count changes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from common.logging import get_logger
from optimizer.annotator import CardinalityAnnotator
from optimizer.optimizer_cost_based import CostBasedOptimizer
from planner.nodes import LogicalNode, LogicalScan
from query_stats.learned import (
    LearnedCardinalityEstimator,
    TableSelectivityModel,
    train_table_model,
)

if TYPE_CHECKING:
    from storage.catalog import Catalog
    from storage.page_store import PageStore

logger = get_logger(__name__)

# Tables larger than this are not trained on the fly (Python-scale engine).
AUTO_TRAIN_MAX_ROWS = 200_000


def ensure_model(
    catalog: Catalog, page_store: PageStore, table_name: str, force: bool = False
) -> TableSelectivityModel | None:
    """Load a fresh model for `table_name`, training one if missing or stale."""
    from storage.statistics import TableStatistics
    from storage.table import TableHandle

    meta = catalog.get_table(table_name)
    model = None if force else TableSelectivityModel.load(catalog.db_dir, table_name.lower())
    if model is not None and model.row_count == meta.row_count:
        return model
    if meta.row_count > AUTO_TRAIN_MAX_ROWS and not force:
        logger.info(f"Skipping auto-training for large table '{table_name}' ({meta.row_count} rows)")
        return None
    handle = TableHandle(meta, page_store, catalog)
    stats = TableStatistics.load(catalog.db_dir, table_name)
    model = train_table_model(handle, stats)
    if model is not None:
        model.save(catalog.db_dir)
    return model


def _scan_tables(node: LogicalNode) -> set[str]:
    out = {node.table_name.lower()} if isinstance(node, LogicalScan) else set()
    for c in node.children:
        out |= _scan_tables(c)
    return out


class LearnedOptimizer(CostBasedOptimizer):
    @property
    def optimizer_id(self) -> str:
        return "learned"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._current_plan_tables: set[str] = set()

    def optimize(self, logical_plan, catalog, page_store, db_name="default"):  # type: ignore[override]
        self._current_plan_tables = _scan_tables(logical_plan)
        return super().optimize(logical_plan, catalog, page_store, db_name)

    def _make_annotator(self, catalog: Catalog, page_store: PageStore, db_name: str) -> CardinalityAnnotator:
        models = {}
        for t in self._current_plan_tables:
            try:
                m = ensure_model(catalog, page_store, t)
            except Exception as exc:  # never fail a query because training failed
                logger.warning(f"Learned model unavailable for '{t}': {exc}")
                m = None
            if m is not None:
                models[t] = m
        return CardinalityAnnotator(catalog, db_name, estimator=LearnedCardinalityEstimator(models))

    def _result_metadata(self, annotator: CardinalityAnnotator) -> dict:
        meta = super()._result_metadata(annotator)
        est = annotator.estimator
        if isinstance(est, LearnedCardinalityEstimator):
            meta["learned_models"] = {
                name: m.metrics for name, m in est.models.items()
            }
            meta["learned_filter_estimates"] = est.hits
            meta["histogram_fallbacks"] = est.fallbacks
        return meta
