"""
optimizer.cost_model — Analytical cost model for query plan evaluation.

Combines I/O costs (sequential vs random page accesses) with CPU costs
(tuple processing, comparison evaluations, hash table building, sorting)
to produce a comparable scalar cost estimate for any physical operator tree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from common.logging import get_logger
from executor.aggregate import HashAggregate
from executor.base import PhysicalOperator
from executor.filter import Filter
from executor.join import HashJoin, NestedLoopJoin
from executor.limit import Limit
from executor.project import Project
from executor.scan import SeqScan
from executor.sort import Sort

logger = get_logger(__name__)


@dataclass(frozen=True)
class CostModelConfig:
    """
    Tunable cost model parameters.

    Documented as empirical baseline weights (see Section J and ADR-005).
    """
    seq_page_cost: float = 1.0       # Cost of sequentially reading one disk page
    random_page_cost: float = 4.0    # Cost of non-sequential page read
    cpu_tuple_cost: float = 0.01      # Cost of processing one tuple
    cpu_operator_cost: float = 0.0025 # Cost of evaluating one predicate/expression
    hash_mem_penalty: float = 0.05    # Overhead per row stored in in-memory hash table
    sort_cost_factor: float = 1.5     # Multiplier for n * log(n) comparison work


class CostModel:
    """
    Calculates the estimated cost of executing a physical plan tree.
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self.config = config or CostModelConfig()

    def estimate_cost(self, op: PhysicalOperator) -> float:
        """
        Recursively compute the total estimated cost of a physical operator tree.
        """
        match op:
            case SeqScan():
                pages = max(1, getattr(op.table_handle, "page_count", 1))
                rows = op.estimated_rows if op.estimated_rows > 0 else getattr(op.table_handle, "row_count", 10)
                io_cost = self.config.seq_page_cost * pages
                cpu_cost = self.config.cpu_tuple_cost * rows
                return io_cost + cpu_cost

            case Filter():
                child_cost = self.estimate_cost(op.child)
                in_rows = op.child.estimated_rows if op.child.estimated_rows > 0 else 10.0
                cpu_cost = self.config.cpu_operator_cost * in_rows
                return child_cost + cpu_cost

            case Project():
                child_cost = self.estimate_cost(op.child)
                in_rows = op.child.estimated_rows if op.child.estimated_rows > 0 else 10.0
                # Cost scales with number of projection expressions
                proj_count = len(op.projections)
                cpu_cost = self.config.cpu_operator_cost * in_rows * max(1, proj_count)
                return child_cost + cpu_cost

            case HashJoin():
                # Right child is build side, Left child is probe side
                build_cost = self.estimate_cost(op.right)
                probe_cost = self.estimate_cost(op.left)
                build_rows = max(1.0, op.right.estimated_rows)
                probe_rows = max(1.0, op.left.estimated_rows)

                # Build phase: inserting each right row into hash table
                build_hash_cost = (self.config.cpu_tuple_cost + self.config.hash_mem_penalty) * build_rows
                # Probe phase: hashing probe key and evaluating match
                probe_hash_cost = self.config.cpu_tuple_cost * probe_rows

                return build_cost + probe_cost + build_hash_cost + probe_hash_cost

            case NestedLoopJoin():
                outer_cost = self.estimate_cost(op.left)
                inner_cost = self.estimate_cost(op.right)
                outer_rows = max(1.0, op.left.estimated_rows)
                inner_rows = max(1.0, op.right.estimated_rows)

                # For each outer row, iterates over inner rows + condition evaluation
                loop_cost = (outer_rows * inner_rows) * self.config.cpu_operator_cost
                return outer_cost + (outer_rows * inner_cost) + loop_cost

            case HashAggregate():
                child_cost = self.estimate_cost(op.child)
                in_rows = max(1.0, op.child.estimated_rows)
                out_rows = max(1.0, op.estimated_rows)

                hash_cost = (self.config.cpu_tuple_cost + self.config.hash_mem_penalty) * out_rows
                group_eval_cost = self.config.cpu_operator_cost * in_rows
                return child_cost + hash_cost + group_eval_cost

            case Sort():
                child_cost = self.estimate_cost(op.child)
                in_rows = max(1.0, op.child.estimated_rows)
                # n * log2(n) comparisons
                n_log_n = in_rows * math.log2(max(in_rows, 2.0))
                sort_cpu_cost = self.config.sort_cost_factor * n_log_n * self.config.cpu_operator_cost
                return child_cost + sort_cpu_cost

            case Limit():
                child_cost = self.estimate_cost(op.child)
                in_rows = max(1.0, op.child.estimated_rows)
                processed_rows = min(in_rows, float(op.limit + (op.offset or 0)))
                return child_cost + (processed_rows * self.config.cpu_operator_cost)

            case _:
                # Fallback: sum children costs
                total = 0.0
                for c in op.children:
                    total += self.estimate_cost(c)
                return total + 1.0
