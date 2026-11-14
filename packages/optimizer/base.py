"""
optimizer.base — BaseOptimizer abstract class for all QueryOpt Lab optimizers.

Every optimizer (Rule-based, Cost-based, Adaptive) must inherit from
BaseOptimizer and produce an OptimizationResult.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from optimizer.result import OptimizationResult
from planner.nodes import LogicalNode

if TYPE_CHECKING:
    from storage.catalog import Catalog
    from storage.page_store import PageStore


class BaseOptimizer(ABC):
    """
    Abstract base class for QueryOpt Lab query optimizers.
    """

    @property
    @abstractmethod
    def optimizer_id(self) -> str:
        """Stable machine-readable identifier for this optimizer."""
        ...

    @abstractmethod
    def optimize(
        self,
        logical_plan: LogicalNode,
        catalog: Catalog,
        page_store: PageStore,
        db_name: str = "default",
    ) -> OptimizationResult:
        """
        Transform the input canonical LogicalNode tree into an optimized
        LogicalNode tree, and compile it to an executable PhysicalOperator.
        """
        ...
