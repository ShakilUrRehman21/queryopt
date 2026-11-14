"""
optimizer.registry — Single place that names and builds every optimizer.

Optimizer ids may carry parameters, which makes ablations a one-line change in
an experiment YAML:

    cost_based                         plain CBO (histograms)
    cost_based[qerr=10,mode=under]     CBO with injected 10x under-estimation
    learned                            CBO + self-supervised learned estimator
    adaptive                           CBO + between-query cardinality feedback
    llm_guided                         LLM proposes the join order; verified, costed
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from optimizer.base import BaseOptimizer


@dataclass(frozen=True)
class OptimizerInfo:
    id: str
    name: str
    family: str  # "baseline" | "classical" | "learning" | "ai"
    description: str

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


OPTIMIZER_INFO: list[OptimizerInfo] = [
    OptimizerInfo("naive", "Naive", "baseline",
                  "Canonical plan straight from the SQL text: no rewrites, original join order."),
    OptimizerInfo("rule_based", "Rule-Based", "classical",
                  "Heuristic rewrites: constant folding, predicate and projection pushdown."),
    OptimizerInfo("cost_based", "Cost-Based", "classical",
                  "System R style DP join enumeration over histogram cardinality estimates."),
    OptimizerInfo("adaptive", "Adaptive (feedback)", "learning",
                  "CBO that replaces estimates with cardinalities observed in earlier executions."),
    OptimizerInfo("learned", "Learned estimator", "learning",
                  "CBO whose filter estimates come from a self-supervised neural model per table."),
    OptimizerInfo("llm_guided", "LLM-guided", "ai",
                  "A language model proposes the join order; the engine validates, costs and runs it."),
]

_SPEC = re.compile(r"^\s*([a-z_]+)\s*(?:\[(.*)\])?\s*$")


def parse_spec(spec: str) -> tuple[str, dict[str, str]]:
    m = _SPEC.match(spec or "")
    if not m:
        raise ValueError(f"Invalid optimizer spec: {spec!r}")
    params: dict[str, str] = {}
    if m.group(2):
        for part in m.group(2).split(","):
            if not part.strip():
                continue
            k, _, v = part.partition("=")
            params[k.strip()] = v.strip()
    return m.group(1), params


def _with_noise(params: dict[str, str]) -> dict[str, Any]:
    """Translate qerr/mode/seed params into an estimator kwarg."""
    if "qerr" not in params:
        return {}
    from query_stats.estimator import HistogramCardinalityEstimator
    from query_stats.noisy import NoisyCardinalityEstimator

    return {
        "estimator": NoisyCardinalityEstimator(
            HistogramCardinalityEstimator(),
            q_error=float(params["qerr"]),
            mode=params.get("mode", "random"),
            seed=int(params.get("seed", 0)),
        )
    }


def create_optimizer(spec: str) -> BaseOptimizer:
    name, params = parse_spec(spec)
    builders: dict[str, Callable[[], BaseOptimizer]] = {}

    def naive() -> BaseOptimizer:
        from optimizer.optimizer_naive import NaiveOptimizer
        return NaiveOptimizer()

    def rule_based() -> BaseOptimizer:
        from optimizer.optimizer_rule_based import RuleBasedOptimizer
        return RuleBasedOptimizer()

    def cost_based() -> BaseOptimizer:
        from optimizer.optimizer_cost_based import CostBasedOptimizer
        return CostBasedOptimizer(**_with_noise(params))

    def adaptive() -> BaseOptimizer:
        from optimizer.optimizer_adaptive import AdaptiveOptimizer
        return AdaptiveOptimizer(**_with_noise(params))

    def learned() -> BaseOptimizer:
        from optimizer.optimizer_learned import LearnedOptimizer
        return LearnedOptimizer()

    def llm_guided() -> BaseOptimizer:
        from ai.llm_optimizer import LLMGuidedOptimizer
        return LLMGuidedOptimizer(**{k: v for k, v in params.items() if k in ("model", "provider")})

    builders.update(naive=naive, rule_based=rule_based, cost_based=cost_based,
                    adaptive=adaptive, learned=learned, llm_guided=llm_guided)
    if name not in builders:
        known = ", ".join(i.id for i in OPTIMIZER_INFO)
        raise ValueError(f"Unknown optimizer '{name}'. Known optimizers: {known}")
    opt = builders[name]()
    if params and name not in ("cost_based", "adaptive", "llm_guided"):
        raise ValueError(f"Optimizer '{name}' takes no parameters (got {params})")
    return opt


def list_optimizers() -> list[dict[str, str]]:
    return [i.to_dict() for i in OPTIMIZER_INFO]
