"""sqlopt — static, dialect-aware SQL optimizer (rules first, optional LLM)."""
from sqlopt.optimizer import DIALECTS, OptimizeResult, optimize_sql

__all__ = ["DIALECTS", "OptimizeResult", "optimize_sql"]
