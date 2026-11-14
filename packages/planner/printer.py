"""
planner.printer — Human-readable text representation of plan trees.

Produces formatted ASCII tree outputs suitable for CLI `EXPLAIN` commands
and log files, similar to Postgres/DuckDB EXPLAIN output.
"""
from __future__ import annotations

from typing import Any
from planner.nodes import LogicalNode


def print_logical_plan(node: LogicalNode, indent: str = "", is_last: bool = True) -> str:
    """
    Format a LogicalPlanTree as a human-readable ASCII hierarchy.

    Example:
    LogicalProject [id, name]
    └── LogicalFilter [id > 10]
        └── LogicalScan [users] (cols: id, name, age)
    """
    lines: list[str] = []
    marker = "\\-- " if is_last else "+-- "
    prefix = indent + marker if indent else ""

    details = node.details()
    detail_str = ""
    if details:
        pairs = [f"{k}={v}" for k, v in details.items()]
        detail_str = f" ({', '.join(pairs)})"

    cost_str = ""
    if node.estimated_rows is not None or node.estimated_cost is not None:
        cost_parts = []
        if node.estimated_rows is not None:
            cost_parts.append(f"rows={node.estimated_rows:.1f}")
        if node.estimated_cost is not None:
            cost_parts.append(f"cost={node.estimated_cost:.2f}")
        cost_str = f" [{', '.join(cost_parts)}]"

    lines.append(f"{prefix}{node.node_name()}{detail_str}{cost_str}")

    next_indent = indent + ("    " if is_last else "|   ")
    children = node.children
    for idx, child in enumerate(children):
        child_is_last = idx == len(children) - 1
        lines.append(print_logical_plan(child, indent=next_indent, is_last=child_is_last))

    return "\n".join(lines)


def explain_logical_plan(node: LogicalNode) -> str:
    """Entry point to format a logical plan."""
    return print_logical_plan(node, indent="", is_last=True)
