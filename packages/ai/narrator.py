"""
ai.narrator — "Why did the optimizer do that?" (Research Gap 3, Persona 4).

Turns an executed plan into

1. **facts** — a structured, verifiable record: each operator with estimated
   vs. actual rows and q-error, the join order and algorithms, the rules that
   fired, and optimizer provenance (feedback corrections, learned-model
   accuracy, LLM rationale);
2. a **deterministic narrative** built only from those facts — always
   available, testable, and identical across runs;
3. optionally an **LLM narrative** — the same facts handed to a language model
   with instructions to use no number that is not in them, for a friendlier
   explanation or to answer a free-form question about the plan.

Keeping (1) separate from (2)/(3) is the point: the UI can show the model's
prose next to the facts it was given, so a reader can check every claim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai.llm import LLMClient, LLMUnavailableError, get_llm_client
from optimizer.signature import canonical_expr

if TYPE_CHECKING:
    from common.types import QueryResult
    from executor.base import PhysicalOperator
    from optimizer.result import OptimizationResult

_MISESTIMATE_Q = 3.0  # q-error above which an estimate is called "wrong"


def q_error(est: float, act: float) -> float:
    e, a = max(float(est), 1.0), max(float(act), 1.0)
    return max(e / a, a / e)


@dataclass
class OperatorFact:
    step: int
    depth: int
    operator: str
    target: str
    estimated_rows: float
    actual_rows: int
    q_error: float
    time_ms: float
    direction: str  # "under" | "over" | "ok"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PlanFacts:
    sql: str
    optimizer_id: str
    status: str
    rows_returned: int
    execution_time_ms: float
    planning_time_ms: float
    estimated_cost: float
    rules_applied: list[str]
    join_order: list[str]
    join_algorithms: list[str]
    operators: list[OperatorFact]
    worst: OperatorFact | None
    metadata: dict[str, Any] = field(default_factory=dict)
    # False for optimizers that never estimate cardinalities (naive, rule_based):
    # their "estimates" are zeros, so q-errors would be meaningless.
    has_estimates: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("operators", "worst")}
        d["operators"] = [o.to_dict() for o in self.operators]
        d["worst"] = self.worst.to_dict() if self.worst else None
        return d


def _target(op: PhysicalOperator) -> str:
    name = op.name
    if name == "SeqScan":
        alias = getattr(op, "alias", None)
        t = getattr(op, "table_name", "?")
        return f"{t}" + (f" AS {alias}" if alias and alias != t else "")
    if name == "Filter":
        return canonical_expr(op.predicate)  # type: ignore[attr-defined]
    if name in ("HashJoin", "NestedLoopJoin"):
        return canonical_expr(op.condition)  # type: ignore[attr-defined]
    if name == "HashAggregate":
        gb = getattr(op, "group_by", None) or []
        return "GROUP BY " + ", ".join(canonical_expr(g) for g in gb) if gb else "scalar aggregate"
    if name == "Sort":
        items = getattr(op, "sort_items", []) or []
        return ", ".join(f"{canonical_expr(i.expr)} {i.direction.name}" for i in items)
    if name == "Limit":
        return f"LIMIT {getattr(op, 'limit', '?')}"
    if name == "Project":
        return f"{len(op.schema.columns)} columns"
    return ""


def collect_facts(
    sql: str,
    opt_result: OptimizationResult,
    result: QueryResult,
) -> PlanFacts:
    root = opt_result.physical_plan
    ops: list[OperatorFact] = []
    join_order: list[str] = []
    join_algos: list[str] = []

    def leaves(op: PhysicalOperator) -> None:
        if op.name == "SeqScan":
            join_order.append(getattr(op, "alias", None) or op.table_name)  # type: ignore[attr-defined]
        for c in op.children:
            leaves(c)

    leaves(root)

    # Post-order = execution order as a reader would describe it.
    def walk(op: PhysicalOperator, depth: int) -> None:
        for c in op.children:
            walk(c, depth + 1)
        if op.name in ("HashJoin", "NestedLoopJoin"):
            join_algos.append(op.name)
        est = float(op.estimated_rows or 0.0)
        act = int(op.actual_rows)
        q = q_error(est, act)
        direction = "ok" if q < _MISESTIMATE_Q else ("under" if est < act else "over")
        ops.append(
            OperatorFact(
                step=len(ops) + 1,
                depth=depth,
                operator=op.name,
                target=_target(op),
                estimated_rows=round(est, 1),
                actual_rows=act,
                q_error=round(q, 2),
                time_ms=round(op.execution_time_ms, 3),
                direction=direction,
            )
        )

    walk(root, 0)
    has_estimates = any(o.estimated_rows > 0 for o in ops)
    if not has_estimates:
        for o in ops:
            o.q_error, o.direction = 1.0, "ok"
    worst = max(ops, key=lambda o: o.q_error) if ops and has_estimates else None
    return PlanFacts(
        sql=sql,
        optimizer_id=opt_result.optimizer_id,
        status=result.status.name,
        rows_returned=result.rows_returned,
        execution_time_ms=round(result.execution_time_ms, 3),
        planning_time_ms=round(result.planning_time_ms, 3),
        estimated_cost=round(opt_result.estimated_cost, 4),
        rules_applied=list(opt_result.rules_applied),
        join_order=join_order if len(join_order) > 1 else [],
        join_algorithms=join_algos,
        operators=ops,
        worst=worst,
        metadata=opt_result.metadata,
        has_estimates=has_estimates,
    )


# ── Deterministic narrative ────────────────────────────────────────────────

_RULE_TEXT = {
    "ConstantFolding": "folded constant expressions at plan time",
    "PredicatePushdown": "pushed filters down so rows are discarded before joins",
    "ProjectionPushdown": "pruned unused columns early",
    "join_reorder_dp": "searched join orders with dynamic programming",
}


def _fmt(n: float) -> str:
    return f"{n:,.0f}" if n >= 10 else f"{n:,.1f}".rstrip("0").rstrip(".")


def _describe_op(o: OperatorFact) -> str:
    match o.operator:
        case "SeqScan":
            s = f"Read every row of **{o.target}** ({_fmt(o.actual_rows)} rows)."
        case "Filter":
            s = f"Keep rows where `{o.target}`"
        case "HashJoin":
            s = f"Hash-join on `{o.target}` (build a hash table on the right input, stream the left)"
        case "NestedLoopJoin":
            s = f"Nested-loop join on `{o.target}` (compare every pair of rows)"
        case "HashAggregate":
            s = f"Aggregate ({o.target})"
        case "Sort":
            s = f"Sort by {o.target}"
        case "Limit":
            s = f"Stop after {o.target.replace('LIMIT ', '')} rows"
        case "Project":
            s = f"Output {o.target.replace('1 columns', '1 column')}"
        case _:
            s = f"{o.operator} {o.target}"
    if o.operator == "SeqScan":
        return s
    if o.estimated_rows <= 0:
        return s + f": {_fmt(o.actual_rows)} rows."
    s += f": estimated {_fmt(o.estimated_rows)}, actually {_fmt(o.actual_rows)} rows"
    if o.direction != "ok":
        s += f" (**{o.q_error:.1f}× {'under' if o.direction == 'under' else 'over'}-estimate**)"
    return s + "."


def narrate(facts: PlanFacts) -> dict[str, Any]:
    """Return {'summary', 'steps', 'why', 'misestimates', 'suggestions'} from facts alone."""
    steps = [f"{o.step}. {_describe_op(o)}" for o in facts.operators]

    why: list[str] = []
    rules = [_RULE_TEXT.get(r, r) for r in dict.fromkeys(facts.rules_applied)]
    if rules:
        why.append(f"The **{facts.optimizer_id}** optimizer " + "; ".join(rules) + ".")
    if facts.join_order:
        why.append(
            "Join order: " + ", then ".join(f"**{t}**" for t in facts.join_order)
            + ". Plans are left-deep: each newly added relation is the hash-join build side, "
            "so the cost model favours adding small (or well-filtered) relations and keeping "
            "intermediate results small."
        )
    if "NestedLoopJoin" in facts.join_algorithms:
        why.append(
            "A nested-loop join was used because the join condition is not a simple equality; "
            "it costs |left| × |right| comparisons."
        )
    meta = facts.metadata or {}
    for c in meta.get("feedback_corrections", [])[:5]:
        why.append(
            f"Feedback replaced the modelled estimate for {', '.join(c['tables'])}"
            + (f" where {' AND '.join(c['predicates'])}" if c["predicates"] else "")
            + f" (modelled {_fmt(c['modelled_rows'])}, observed {_fmt(c['observed_rows'])} rows)."
        )
    if "learned_models" in meta:
        for t, m in meta["learned_models"].items():
            if "histogram_p95_qerror" in m:
                why.append(
                    f"Filters on **{t}** were estimated by a learned model "
                    f"(held-out p95 q-error {m['learned_p95_qerror']} vs. {m['histogram_p95_qerror']} for histograms)."
                )
    llm = meta.get("llm")
    if llm:
        if llm.get("used"):
            why.append(
                f"The join order was proposed by {llm.get('model')}"
                + (f": “{llm['rationale']}”" if llm.get("rationale") else ".")
            )
            if llm.get("estimated_cost_ratio_vs_dp"):
                why.append(
                    f"By the cost model, that order costs {llm['estimated_cost_ratio_vs_dp']:.2f}× "
                    "the dynamic-programming optimum."
                )
        elif llm.get("fallback_reason"):
            why.append(f"LLM guidance was not used: {llm['fallback_reason']}.")

    # Project/Sort pass their input through unchanged, so they only echo the
    # error of the operator below them; don't report it twice.
    bad = [o for o in facts.operators if o.direction != "ok" and o.operator not in ("Project", "Sort")]
    misest = [
        f"`{o.operator}` on `{o.target}`: estimated {_fmt(o.estimated_rows)}, actual {_fmt(o.actual_rows)} "
        f"(q-error {o.q_error:.1f})"
        for o in sorted(bad, key=lambda o: -o.q_error)[:5]
    ]

    suggestions: list[str] = []
    multi_conj_filter = any(o.operator == "Filter" and "AND" in o.target and o.direction != "ok" for o in bad)
    if multi_conj_filter:
        suggestions.append(
            "A conjunctive filter was mis-estimated — the columns are probably correlated. "
            "Try the **learned** optimizer, which models predicates jointly instead of multiplying them."
        )
    if bad and facts.optimizer_id != "adaptive":
        suggestions.append(
            "Run this query once more with the **adaptive** optimizer: it will reuse the cardinalities "
            "just observed instead of the estimates above."
        )
    if "NestedLoopJoin" in facts.join_algorithms:
        suggestions.append("Rewrite the join condition as an equality so a hash join can be used.")
    if not facts.has_estimates:
        suggestions.append(
            f"The {facts.optimizer_id} optimizer does not estimate cardinalities. Compare it with "
            "cost_based, adaptive or learned in the Arena to see what statistics change."
        )
    elif not bad:
        suggestions.append("Estimates were within 3× everywhere; this plan is well-informed.")

    summary = (
        f"{facts.rows_returned:,} rows in {facts.execution_time_ms:.1f} ms "
        f"(+{facts.planning_time_ms:.1f} ms planning) using the {facts.optimizer_id} optimizer. "
        + (
            f"The largest estimation error was {facts.worst.q_error:.1f}× at step {facts.worst.step} "
            f"({facts.worst.operator})."
            if facts.worst and facts.worst.q_error >= _MISESTIMATE_Q
            else (
                "All cardinality estimates were within 3× of reality."
                if facts.has_estimates
                else "This optimizer does not estimate cardinalities."
            )
        )
    )
    return {
        "summary": summary,
        "steps": steps,
        "why": why,
        "misestimates": misest,
        "suggestions": suggestions,
    }


# ── LLM narrative ─────────────────────────────────────────────────────────

NARRATOR_SYSTEM = """You are the explanation layer of QueryOpt Lab, a research database.
You receive FACTS (JSON) measured by the engine for one executed query plan.
Rules:
- Use only numbers that appear in FACTS. Never invent row counts, timings or costs.
- Refer to plan steps by their step number.
- Be concrete and concise; prefer short paragraphs and bullet points (Markdown).
- If FACTS cannot answer the question, say so plainly.
- Audience: a curious developer who knows SQL but not optimizer internals."""


def llm_narrate(
    facts: PlanFacts,
    question: str | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    client = client or get_llm_client()
    if not client.available:
        raise LLMUnavailableError("No language model configured.")
    task = question.strip() if question else (
        "Explain how this plan executes, why the optimizer chose it, where its estimates "
        "were wrong and what the consequences were. Finish with one actionable suggestion."
    )
    prompt = (
        f"SQL:\n{facts.sql}\n\nFACTS:\n{json.dumps(facts.to_dict(), indent=1, default=str)}\n\n"
        f"TASK: {task}"
    )
    resp = client.complete(NARRATOR_SYSTEM, prompt, max_tokens=900)
    return {"text": resp.text.strip(), **resp.provenance()}
