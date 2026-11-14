"""
ai — AI-assisted query optimization for QueryOpt Lab.

Modules
-------
llm               Provider-agnostic LLM client (Claude / OpenAI-compatible / offline) + replay cache
context           Factual schema/statistics descriptions for prompts
narrator          Fact-grounded plan explanations (deterministic + optional LLM prose)
llm_optimizer     LLM-proposed join orders, validated and costed by the engine (RQ5)
nl2sql            Natural language → SQL with an engine-in-the-loop repair cycle
rewrite_verifier  Empirical equivalence + speed checks for suggested rewrites

Learning components that need no LLM live with the engine:
optimizer.optimizer_adaptive (feedback) and query_stats.learned (neural estimator).
"""
from ai.llm import LLMUnavailableError, get_llm_client

__all__ = ["LLMUnavailableError", "get_llm_client"]
