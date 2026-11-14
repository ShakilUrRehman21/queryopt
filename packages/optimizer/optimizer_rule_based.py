"""
optimizer.optimizer_rule_based — Rule-Based Optimizer (RBO) baseline.

Applies a sequence of heuristic rewrite rules (constant folding,
predicate pushdown, column pruning) iteratively until a fixpoint is reached.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from optimizer.base import BaseOptimizer
from optimizer.physical_planner import PhysicalPlanner
from optimizer.result import OptimizationResult
from optimizer.rules.base import Rule
from optimizer.rules.constant_folding import ConstantFoldingRule
from optimizer.rules.predicate_pushdown import PredicatePushdownRule
from optimizer.rules.projection_pushdown import ProjectionPushdownRule
from planner.nodes import LogicalNode

if TYPE_CHECKING:
    from storage.catalog import Catalog
    from storage.page_store import PageStore


class RuleBasedOptimizer(BaseOptimizer):
    """
    Heuristic rule-based query optimizer.
    """

    def __init__(self, rules: list[Rule] | None = None, max_iterations: int = 10) -> None:
        self.rules = rules or [
            ConstantFoldingRule(),
            PredicatePushdownRule(),
            ProjectionPushdownRule(),
        ]
        self.max_iterations = max_iterations

    @property
    def optimizer_id(self) -> str:
        return "rule_based"

    def optimize(
        self,
        logical_plan: LogicalNode,
        catalog: Catalog,
        page_store: PageStore,
        db_name: str = "default",
    ) -> OptimizationResult:
        start_time = time.perf_counter()
        current_plan = logical_plan
        applied_rules: list[str] = []

        # Iterate rules until fixpoint or max iterations reached
        for _ in range(self.max_iterations):
            any_modified = False
            for rule in self.rules:
                new_plan, modified = rule.apply(current_plan)
                if modified:
                    current_plan = new_plan
                    applied_rules.append(rule.name)
                    any_modified = True
            if not any_modified:
                break

        # Compile to physical execution tree
        physical_planner = PhysicalPlanner(catalog, page_store, db_name)
        physical_plan = physical_planner.plan(current_plan)

        opt_time_ms = (time.perf_counter() - start_time) * 1000.0

        return OptimizationResult(
            optimizer_id=self.optimizer_id,
            initial_logical_plan=logical_plan,
            optimized_logical_plan=current_plan,
            physical_plan=physical_plan,
            optimization_time_ms=opt_time_ms,
            rules_applied=applied_rules,
            alternatives_considered=1,
        )
