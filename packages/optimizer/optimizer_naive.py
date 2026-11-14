"""
optimizer.optimizer_naive — Baseline unoptimized optimizer.

Directly compiles the canonical logical plan to physical execution without
applying any rewrite rules or cost-based reordering.
Used as the reference oracle for optimization correctness testing (Section 44).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from optimizer.base import BaseOptimizer
from optimizer.physical_planner import PhysicalPlanner
from optimizer.result import OptimizationResult
from planner.nodes import LogicalNode

if TYPE_CHECKING:
    from storage.catalog import Catalog
    from storage.page_store import PageStore


class NaiveOptimizer(BaseOptimizer):
    """
    Unoptimized baseline optimizer.
    """

    @property
    def optimizer_id(self) -> str:
        return "naive"

    def optimize(
        self,
        logical_plan: LogicalNode,
        catalog: Catalog,
        page_store: PageStore,
        db_name: str = "default",
    ) -> OptimizationResult:
        start_time = time.perf_counter()
        physical_planner = PhysicalPlanner(catalog, page_store, db_name)
        physical_plan = physical_planner.plan(logical_plan)
        opt_time_ms = (time.perf_counter() - start_time) * 1000.0

        return OptimizationResult(
            optimizer_id=self.optimizer_id,
            initial_logical_plan=logical_plan,
            optimized_logical_plan=logical_plan,
            physical_plan=physical_plan,
            optimization_time_ms=opt_time_ms,
            rules_applied=[],
            alternatives_considered=1,
        )
