"""
parser — SQL parsing layer for QueryOpt Lab.

Wraps SQLGlot to produce an internal AST representation that the
planner can consume without knowing about SQLGlot's API.

The adapter pattern here is intentional: if SQLGlot is ever replaced,
only this package needs to change.
"""
