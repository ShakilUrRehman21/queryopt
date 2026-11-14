"""
optimizer.result — OptimizationResult data structure.

Encapsulates all metadata and plans generated during the optimization phase
for visualization, comparison, and execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from executor.base import PhysicalOperator
    from planner.nodes import LogicalNode


@dataclass
class OptimizationResult:
    """
    Result returned by any QueryOpt Lab optimizer.
    """
    optimizer_id: str
    initial_logical_plan: LogicalNode
    optimized_logical_plan: LogicalNode
    physical_plan: PhysicalOperator

    optimization_time_ms: float = 0.0
    estimated_cost: float = 0.0
    rules_applied: list[str] = field(default_factory=list)
    alternatives_considered: int = 1
    # Free-form, JSON-serialisable provenance: which estimator was used,
    # feedback hits, LLM model + rationale, etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_rows(self) -> float | None:
        return getattr(self.optimized_logical_plan, "estimated_rows", None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize optimization result for JSON responses and logging."""
        return {
            "optimizer_id": self.optimizer_id,
            "optimization_time_ms": self.optimization_time_ms,
            "estimated_cost": self.estimated_cost,
            "rules_applied": self.rules_applied,
            "alternatives_considered": self.alternatives_considered,
            "metadata": self.metadata,
            "initial_plan": self.initial_logical_plan.to_dict(),
            "optimized_plan": self.optimized_logical_plan.to_dict(),
        }
