"""
query_stats.learned — Self-supervised learned cardinality estimation.

A small multi-layer perceptron per table predicts the selectivity of a
conjunction of range/equality predicates over that table's numeric columns.
It is trained *without any query log*: the trainer samples thousands of random
conjunctive predicates, labels them with their true selectivity by evaluating
them against the table (vectorised NumPy), and fits the network.

Why this belongs in QueryOpt Lab
--------------------------------
Histograms model each column on its own and combine them with the attribute
independence assumption (P(A ∧ B) = P(A)·P(B)). On correlated data (W8/W9) that
assumption produces order-of-magnitude errors. A model that sees whole
conjunctions learns the joint distribution — the idea behind MSCN (Kipf et al.,
CIDR 2019) and Naru (Yang et al., VLDB 2020) — reduced here to ~200 lines of
dependency-free NumPy so it can be read, modified, and ablated.

Featurisation (per numeric column c, 4 values):
    [lo_c, hi_c, active_c, is_equality_c]    with lo/hi min-max normalised to [0, 1]

Target: log((count + 1) / (N + 1)), trained with MSE (≈ minimising log q-error).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from common.logging import get_logger
from common.types import DataType
from parser.ast_types import BinaryOp, ColumnRef, Expr, Literal
from query_stats.estimator import CardinalityEstimate, HistogramCardinalityEstimator

if TYPE_CHECKING:
    from storage.statistics import TableStatistics
    from storage.table import TableHandle

logger = get_logger(__name__)

MODEL_DIRNAME = "learned_models"
_NUMERIC = {DataType.INTEGER, DataType.FLOAT}
_RANGE_OPS = {"<", "<=", ">", ">=", "="}
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "="}


# ── Tiny NumPy MLP ─────────────────────────────────────────────────────────


@dataclass
class MLP:
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    @classmethod
    def init(cls, sizes: list[int], rng: np.random.Generator) -> MLP:
        ws, bs = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            ws.append(rng.normal(0.0, math.sqrt(2.0 / a), size=(a, b)))
            bs.append(np.zeros(b))
        return cls(ws, bs)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        acts = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return h[:, 0], acts

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        epochs: int = 250,
        batch: int = 128,
        lr: float = 3e-3,
    ) -> list[float]:
        """Adam + MSE. Returns per-epoch training loss."""
        m = [np.zeros_like(p) for p in self.weights + self.biases]
        v = [np.zeros_like(p) for p in self.weights + self.biases]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        losses = []
        n = len(x)
        for _ in range(epochs):
            order = rng.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, batch):
                idx = order[s : s + batch]
                xb, yb = x[idx], y[idx]
                pred, acts = self.forward(xb)
                err = pred - yb
                ep_loss += float((err**2).sum())
                # backprop
                grad = (2.0 / len(idx)) * err[:, None]
                gw, gb = [], []
                for i in range(len(self.weights) - 1, -1, -1):
                    gw.insert(0, acts[i].T @ grad)
                    gb.insert(0, grad.sum(axis=0))
                    if i > 0:
                        grad = (grad @ self.weights[i].T) * (acts[i] > 0)
                t += 1
                params = self.weights + self.biases
                grads = gw + gb
                for j, (p, g) in enumerate(zip(params, grads)):
                    m[j] = b1 * m[j] + (1 - b1) * g
                    v[j] = b2 * v[j] + (1 - b2) * g * g
                    mh = m[j] / (1 - b1**t)
                    vh = v[j] / (1 - b2**t)
                    p -= lr * mh / (np.sqrt(vh) + eps)
            losses.append(ep_loss / n)
        return losses


# ── Per-table model ────────────────────────────────────────────────────────


@dataclass
class TableSelectivityModel:
    table: str
    row_count: int
    columns: list[str]
    mins: list[float]
    maxs: list[float]
    integer_columns: list[str]
    mlp: MLP
    metrics: dict[str, Any] = field(default_factory=dict)

    # featurisation ---------------------------------------------------------
    def _norm(self, i: int, v: float) -> float:
        lo, hi = self.mins[i], self.maxs[i]
        return float(np.clip((v - lo) / max(hi - lo, 1e-9), -0.05, 1.05))

    def featurize(self, ranges: dict[str, tuple[float, float, bool]]) -> np.ndarray:
        """ranges: column → (lo, hi, is_equality); use ±inf for open bounds."""
        feats = []
        for i, c in enumerate(self.columns):
            if c in ranges:
                lo, hi, eq = ranges[c]
                lo_n = 0.0 if lo == -math.inf else self._norm(i, lo)
                hi_n = 1.0 if hi == math.inf else self._norm(i, hi)
                feats += [lo_n, hi_n, 1.0, 1.0 if eq else 0.0]
            else:
                feats += [0.0, 1.0, 0.0, 0.0]
        return np.array(feats, dtype=np.float64)

    def selectivity(self, ranges: dict[str, tuple[float, float, bool]]) -> float:
        y = float(self.mlp.predict(self.featurize(ranges)[None, :])[0])
        count = math.exp(y) * (self.row_count + 1) - 1
        return float(np.clip(count / max(self.row_count, 1), 0.0, 1.0))

    # persistence -------------------------------------------------------------
    def save(self, db_dir: Path) -> None:
        d = Path(db_dir) / MODEL_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        arrays = {f"w{i}": w for i, w in enumerate(self.mlp.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.mlp.biases)})
        np.savez(d / f"{self.table}.npz", **arrays)
        meta = {
            "table": self.table,
            "row_count": self.row_count,
            "columns": self.columns,
            "mins": self.mins,
            "maxs": self.maxs,
            "integer_columns": self.integer_columns,
            "layers": len(self.mlp.weights),
            "metrics": self.metrics,
        }
        (d / f"{self.table}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, db_dir: Path, table: str) -> TableSelectivityModel | None:
        d = Path(db_dir) / MODEL_DIRNAME
        meta_p, arr_p = d / f"{table}.json", d / f"{table}.npz"
        if not meta_p.exists() or not arr_p.exists():
            return None
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        arrs = np.load(arr_p)
        n = meta["layers"]
        mlp = MLP([arrs[f"w{i}"] for i in range(n)], [arrs[f"b{i}"] for i in range(n)])
        return cls(
            table=meta["table"],
            row_count=meta["row_count"],
            columns=meta["columns"],
            mins=meta["mins"],
            maxs=meta["maxs"],
            integer_columns=meta.get("integer_columns", []),
            mlp=mlp,
            metrics=meta.get("metrics", {}),
        )


# ── Training ───────────────────────────────────────────────────────────────


def _q_error(est: np.ndarray, act: np.ndarray) -> np.ndarray:
    e = np.maximum(est, 1.0)
    a = np.maximum(act, 1.0)
    return np.maximum(e / a, a / e)


def _sample_workload(
    data: np.ndarray, columns: list[str], int_cols: set[str], n: int, rng: np.random.Generator
) -> list[dict[str, tuple[float, float, bool]]]:
    """Random conjunctions of 1–3 predicates with bounds drawn from real rows."""
    rows, ncol = data.shape
    out = []
    for _ in range(n):
        k = int(rng.integers(1, min(3, ncol) + 1))
        cols = rng.choice(ncol, size=k, replace=False)
        q: dict[str, tuple[float, float, bool]] = {}
        for ci in cols:
            name = columns[ci]
            col = data[:, ci]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                continue
            kind = rng.choice(["lt", "gt", "between", "eq"], p=[0.3, 0.3, 0.3, 0.1])
            a, b = float(rng.choice(col)), float(rng.choice(col))
            if kind == "eq" and name in int_cols:
                q[name] = (a, a, True)
            elif kind == "lt":
                q[name] = (-math.inf, a, False)
            elif kind == "gt":
                q[name] = (a, math.inf, False)
            else:
                q[name] = (min(a, b), max(a, b), False)
        if q:
            out.append(q)
    return out


def _true_counts(
    data: np.ndarray, columns: list[str], queries: list[dict[str, tuple[float, float, bool]]]
) -> np.ndarray:
    idx = {c: i for i, c in enumerate(columns)}
    counts = np.empty(len(queries))
    for qi, q in enumerate(queries):
        mask = np.ones(len(data), dtype=bool)
        for c, (lo, hi, _eq) in q.items():
            col = data[:, idx[c]]
            mask &= (col >= lo) & (col <= hi)
        counts[qi] = mask.sum()
    return counts


def _ranges_to_expr(table: str, q: dict[str, tuple[float, float, bool]]) -> Expr:
    """Build the equivalent predicate so baselines can be scored on the same queries."""
    terms: list[Expr] = []
    for c, (lo, hi, eq) in q.items():
        ref = ColumnRef(column=c, table=table)
        if eq:
            terms.append(BinaryOp(ref, "=", Literal(lo, DataType.FLOAT)))
            continue
        if lo != -math.inf:
            terms.append(BinaryOp(ref, ">=", Literal(lo, DataType.FLOAT)))
        if hi != math.inf:
            terms.append(BinaryOp(ref, "<=", Literal(hi, DataType.FLOAT)))
    expr = terms[0]
    for t in terms[1:]:
        expr = BinaryOp(expr, "AND", t)
    return expr


def train_table_model(
    table: TableHandle,
    table_stats: TableStatistics | None = None,
    n_queries: int = 3000,
    epochs: int = 250,
    seed: int = 7,
) -> TableSelectivityModel | None:
    """Train (and return) a selectivity model for one table's numeric columns."""
    t0 = time.perf_counter()
    schema = table.schema
    num_cols = [c for c in schema.columns if c.data_type in _NUMERIC]
    if not num_cols:
        return None
    col_idx = [schema.column_index(c.name) for c in num_cols]
    rows = list(table.scan())
    if len(rows) < 10:
        return None
    data = np.array(
        [[np.nan if r[i] is None else float(r[i]) for i in col_idx] for r in rows], dtype=np.float64
    )
    columns = [c.name.lower() for c in num_cols]
    int_cols = {c.name.lower() for c in num_cols if c.data_type == DataType.INTEGER}
    rng = np.random.default_rng(seed)

    queries = _sample_workload(data, columns, int_cols, n_queries, rng)
    counts = _true_counts(data, columns, queries)
    n_rows = len(rows)

    model = TableSelectivityModel(
        table=table.name.lower(),
        row_count=n_rows,
        columns=columns,
        mins=[float(np.nanmin(data[:, i])) for i in range(len(columns))],
        maxs=[float(np.nanmax(data[:, i])) for i in range(len(columns))],
        integer_columns=sorted(int_cols),
        mlp=MLP.init([4 * len(columns), 64, 64, 1], rng),
    )
    x = np.stack([model.featurize(q) for q in queries])
    y = np.log((counts + 1.0) / (n_rows + 1.0))

    split = int(0.8 * len(x))
    losses = model.mlp.fit(x[:split], y[:split], rng, epochs=epochs)

    # Held-out evaluation vs. the histogram baseline on identical predicates.
    test_q, test_counts = queries[split:], counts[split:]
    learned_est = np.array([model.selectivity(q) * n_rows for q in test_q])
    learned_qerr = _q_error(learned_est, test_counts)
    metrics: dict[str, Any] = {
        "train_queries": split,
        "test_queries": len(test_q),
        "final_train_loss": round(losses[-1], 5),
        "learned_median_qerror": round(float(np.median(learned_qerr)), 3),
        "learned_p95_qerror": round(float(np.percentile(learned_qerr, 95)), 3),
        "learned_max_qerror": round(float(learned_qerr.max()), 3),
    }
    if table_stats is not None:
        hist = HistogramCardinalityEstimator()
        stats_map = {model.table: table_stats}
        hist_est = np.array(
            [
                hist.estimate_filter(n_rows, _ranges_to_expr(model.table, q), stats_map).estimated_rows
                for q in test_q
            ]
        )
        hist_qerr = _q_error(hist_est, test_counts)
        metrics.update(
            {
                "histogram_median_qerror": round(float(np.median(hist_qerr)), 3),
                "histogram_p95_qerror": round(float(np.percentile(hist_qerr, 95)), 3),
                "histogram_max_qerror": round(float(hist_qerr.max()), 3),
            }
        )
    metrics["training_seconds"] = round(time.perf_counter() - t0, 2)
    model.metrics = metrics
    logger.info(f"Learned estimator for '{model.table}': {metrics}")
    return model


# ── Estimator ──────────────────────────────────────────────────────────────


class LearnedCardinalityEstimator(HistogramCardinalityEstimator):
    """
    Uses a per-table learned model for the numeric conjuncts of a filter and
    the histogram estimator for everything else (text predicates, ORs, joins).
    """

    def __init__(self, models: dict[str, TableSelectivityModel], **kw: Any) -> None:
        super().__init__(**kw)
        self.models = {k.lower(): v for k, v in models.items()}
        self.hits = 0
        self.fallbacks = 0

    def _resolve_table(
        self, ref: ColumnRef, table_stats: dict[str, TableStatistics] | None
    ) -> TableSelectivityModel | None:
        if ref.table:
            key = ref.table.lower()
            if key in self.models:
                return self.models[key]
            if table_stats and key in table_stats:
                return self.models.get(table_stats[key].table_name.lower())
            return None
        col = ref.column.lower()
        candidates = [m for m in self.models.values() if col in m.columns]
        if table_stats:
            present = {ts.table_name.lower() for ts in table_stats.values()}
            candidates = [m for m in candidates if m.table in present]
        return candidates[0] if len(candidates) == 1 else None

    def estimate_filter(
        self,
        input_rows: float,
        predicate: Expr,
        table_stats: dict[str, TableStatistics] | None = None,
    ) -> CardinalityEstimate:
        from optimizer.signature import split_conjuncts

        conjuncts = split_conjuncts(predicate)
        model: TableSelectivityModel | None = None
        ranges: dict[str, list[float]] = {}
        eq_cols: set[str] = set()
        rest: list[Expr] = []

        for c in conjuncts:
            parsed = self._as_range(c)
            if parsed is None:
                rest.append(c)
                continue
            ref, op, val = parsed
            m = self._resolve_table(ref, table_stats)
            col = ref.column.lower()
            if m is None or col not in m.columns or (model is not None and m is not model):
                rest.append(c)
                continue
            model = m
            lo, hi = ranges.get(col, [-math.inf, math.inf])
            if op == "=":
                lo, hi = max(lo, val), min(hi, val)
                eq_cols.add(col)
            elif op in (">", ">="):
                lo = max(lo, val)
            else:
                hi = min(hi, val)
            ranges[col] = [lo, hi]

        if model is None:
            self.fallbacks += 1
            return super().estimate_filter(input_rows, predicate, table_stats)

        self.hits += 1
        sel = model.selectivity({c: (lo, hi, c in eq_cols) for c, (lo, hi) in ranges.items()})
        for r in rest:  # independence only for the parts the model can't see
            sel *= max(0.0, min(1.0, self._compute_selectivity(r, table_stats)))
        est_rows = max(1.0, input_rows * sel) if input_rows > 0 else 0.0
        return CardinalityEstimate(estimated_rows=est_rows, selectivity=sel)

    @staticmethod
    def _as_range(expr: Expr) -> tuple[ColumnRef, str, float] | None:
        if not isinstance(expr, BinaryOp) or expr.op not in _RANGE_OPS:
            return None
        if isinstance(expr.left, ColumnRef) and isinstance(expr.right, Literal):
            ref, lit, op = expr.left, expr.right, expr.op
        elif isinstance(expr.right, ColumnRef) and isinstance(expr.left, Literal):
            ref, lit, op = expr.right, expr.left, _FLIP[expr.op]
        else:
            return None
        if isinstance(lit.value, bool) or not isinstance(lit.value, (int, float)):
            return None
        return ref, op, float(lit.value)
