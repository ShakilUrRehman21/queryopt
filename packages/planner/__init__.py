"""
planner — Logical query planning layer for QueryOpt Lab.

Transforms internal AST into an operator tree of LogicalNode instances.
The logical plan represents WHAT data to retrieve and transform, independent
of specific physical execution algorithms (SeqScan vs IndexScan, HashJoin vs NestedLoopJoin).
"""
