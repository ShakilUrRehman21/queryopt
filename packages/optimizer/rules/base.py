"""
optimizer.rules — Query rewriting optimization rules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from planner.nodes import LogicalNode


class Rule(ABC):
    """
    Abstract base class for a logical plan transformation rule.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this optimization rule."""
        ...

    @abstractmethod
    def apply(self, node: LogicalNode) -> tuple[LogicalNode, bool]:
        """
        Apply the transformation rule to a node and its descendants.
        Returns:
            (new_node, modified): The transformed node and a boolean indicating
                                  if any transformation was performed.
        """
        ...
