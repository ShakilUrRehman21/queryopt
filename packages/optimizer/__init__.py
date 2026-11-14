"""
optimizer — Query optimization framework for QueryOpt Lab.

Provides:
- BaseOptimizer interface
- Rule-based query rewriting (RuleBasedOptimizer)
- Cost-based plan generation and search (CostBasedOptimizer)
- Cardinality annotation (CardinalityAnnotator)
- Cost modeling (CostModel, CostModelConfig)
- Baseline unoptimized reference (NaiveOptimizer)
"""
from optimizer.annotator import CardinalityAnnotator
from optimizer.base import BaseOptimizer
from optimizer.cost_model import CostModel, CostModelConfig
from optimizer.optimizer_cost_based import CostBasedOptimizer
from optimizer.optimizer_naive import NaiveOptimizer
from optimizer.optimizer_rule_based import RuleBasedOptimizer
from optimizer.result import OptimizationResult

__all__ = [
    "BaseOptimizer",
    "CardinalityAnnotator",
    "CostBasedOptimizer",
    "CostModel",
    "CostModelConfig",
    "NaiveOptimizer",
    "OptimizationResult",
    "RuleBasedOptimizer",
]
