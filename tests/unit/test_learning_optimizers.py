"""
Tests for the Phase 6 learning components: signatures, cardinality feedback,
the adaptive optimizer, the learned estimator, error injection and the registry.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from benchmark.generator import DatasetGenerator
from benchmark.workloads import create_workload
from common.types import DataType
from optimizer.registry import create_optimizer, parse_spec
from optimizer.signature import RelSignature, canonical_expr, logical_signature
from parser.ast_types import BinaryOp, ColumnRef, Literal
from queryopt.engine import QueryEngine

W8_Q1 = "SELECT id, temperature, humidity FROM sensors WHERE temperature > 85 AND humidity > 85"
W9_Q1 = create_workload("W9").queries[0].sql
W9_Q2 = create_workload("W9").queries[1].sql


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    d = tmp_path_factory.mktemp("learning")
    eng = QueryEngine(data_dir=d, db_name="db")
    DatasetGenerator(seed=11).populate_database(eng.db, create_workload("W9").table_specs)
    return eng


def _root_filter(result):
    return next(s for s in result.operator_stats.values() if s.operator_type == "Filter")


# ── signatures ────────────────────────────────────────────────────────────


def test_canonical_expr_normalises_commutativity_and_flips():
    a = BinaryOp(ColumnRef("x", "t"), "=", ColumnRef("y", "u"))
    b = BinaryOp(ColumnRef("y", "u"), "=", ColumnRef("x", "t"))
    assert canonical_expr(a) == canonical_expr(b)
    lt = BinaryOp(Literal(5, DataType.INTEGER), "<", ColumnRef("x", "t"))
    gt = BinaryOp(ColumnRef("x", "t"), ">", Literal(5, DataType.INTEGER))
    assert canonical_expr(lt) == canonical_expr(gt)


def test_signature_is_join_order_independent(engine):
    a = engine.optimize_select(W9_Q1, "cost_based").optimized_logical_plan
    b = engine.optimize_select(W9_Q1, "naive").optimized_logical_plan
    sa, sb = logical_signature(a), logical_signature(b)
    assert sa is not None and sb is not None
    assert sa.tables == sb.tables == frozenset({"alerts", "sensors", "zones"})
    assert sa.key == sb.key


# ── feedback + adaptive ──────────────────────────────────────────────────


def test_feedback_recorded_and_used_by_adaptive(engine):
    engine.feedback_store.reset()
    cold = engine.execute(W8_Q1, "cost_based")
    f_cold = _root_filter(cold)
    assert f_cold.actual_rows > 3 * f_cold.estimated_rows  # correlation → under-estimate
    assert len(engine.feedback_store) > 0

    warm = engine.execute(W8_Q1, "adaptive")
    f_warm = _root_filter(warm)
    assert f_warm.estimated_rows == pytest.approx(f_warm.actual_rows)
    corrections = warm.optimizer_metadata["feedback_corrections"]
    assert corrections and corrections[0]["observed_rows"] == f_warm.actual_rows


def test_adaptive_corrects_join_subsets_it_never_ran_in_that_order(engine):
    engine.feedback_store.reset()
    engine.execute(W9_Q1, "rule_based")  # different join order, same sub-expressions
    res = engine.optimize_select(W9_Q1, "adaptive")
    assert len(res.metadata["feedback_corrections"]) >= 2


def test_feedback_reset(engine):
    engine.execute(W8_Q1, "cost_based")
    engine.feedback_store.reset()
    assert len(engine.feedback_store) == 0
    assert engine.optimize_select(W8_Q1, "adaptive").metadata["feedback_corrections"] == []


def test_limit_subtrees_are_not_recorded(engine):
    engine.feedback_store.reset()
    engine.execute("SELECT id FROM sensors WHERE temperature > 10 LIMIT 3", "cost_based")
    assert len(engine.feedback_store) == 0


# ── learned estimator ────────────────────────────────────────────────────


def test_learned_model_beats_histograms_on_correlated_table(engine):
    from query_stats.learned import train_table_model
    from storage.statistics import TableStatistics

    handle = engine.db.get_table("sensors")
    stats = TableStatistics.load(engine.db.db_dir, "sensors")
    model = train_table_model(handle, stats, n_queries=1200, epochs=120, seed=3)
    m = model.metrics
    assert m["learned_p95_qerror"] < m["histogram_p95_qerror"]
    assert m["learned_median_qerror"] < 2.0


def test_learned_optimizer_uses_model_for_filters(engine):
    # Per-query accuracy varies; aggregate accuracy is tested above on held-out queries.
    learned = engine.execute(W8_Q1, "learned")
    assert learned.status.name == "SUCCESS"
    assert "sensors" in learned.optimizer_metadata["learned_models"]
    assert learned.optimizer_metadata["learned_filter_estimates"] >= 1


# ── error injection ──────────────────────────────────────────────────────


@pytest.mark.parametrize("mode,check", [("under", lambda b, n: n < b), ("over", lambda b, n: n > b)])
def test_error_injection_modes(engine, mode, check):
    base = _root_filter(engine.execute(W8_Q1, "cost_based")).estimated_rows
    noisy = _root_filter(engine.execute(W8_Q1, f"cost_based[qerr=10,mode={mode}]")).estimated_rows
    assert check(base, noisy)


def test_random_error_injection_is_deterministic(engine):
    a = _root_filter(engine.execute(W8_Q1, "cost_based[qerr=10,seed=4]")).estimated_rows
    b = _root_filter(engine.execute(W8_Q1, "cost_based[qerr=10,seed=4]")).estimated_rows
    assert a == b


# ── registry ─────────────────────────────────────────────────────────────


def test_parse_spec():
    assert parse_spec("cost_based[qerr=10, mode=under]") == ("cost_based", {"qerr": "10", "mode": "under"})
    assert parse_spec("adaptive") == ("adaptive", {})


def test_unknown_optimizer_is_an_error_not_a_fallback(engine):
    with pytest.raises(ValueError):
        create_optimizer("quantum")
    res = engine.execute("SELECT id FROM zones", "quantum")
    assert res.status.name == "ERROR" and "Unknown optimizer" in res.error_message


# ── oracle: every optimizer returns the same rows ───────────────────────


@pytest.mark.parametrize("sql", [W8_Q1, W9_Q1, W9_Q2])
def test_all_optimizers_agree(engine, sql):
    rows = None
    for spec in ["naive", "rule_based", "cost_based", "adaptive", "learned", "llm_guided",
                 "cost_based[qerr=100,mode=over]"]:
        res = engine.execute(sql, spec)
        assert res.status.name == "SUCCESS", (spec, res.error_message)
        got = sorted(res.rows, key=repr)
        rows = got if rows is None else rows
        assert got == rows, spec


def test_conjunctive_predicate_is_split_and_pushed_below_joins(engine):
    plan = engine.optimize_select(W9_Q2, "rule_based").physical_plan
    assert plan.name != "Filter" and getattr(plan, "child", plan).name != "Filter"


# ── data generation ──────────────────────────────────────────────────────


def test_generator_is_stable_across_processes():
    code = (
        "import sys; sys.path.insert(0,'packages');"
        "from benchmark.generator import DatasetGenerator;"
        "from benchmark.workloads import get_standard_tables;"
        "print(DatasetGenerator(5).generate_table(get_standard_tables()[1])[:3])"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": str(seed), "PATH": ""}).stdout
        for seed in (1, 2)
    }
    assert len(outs) == 1 and next(iter(outs)).strip()


def test_correlated_columns_are_correlated():
    from benchmark.workloads import get_correlated_tables

    rows = DatasetGenerator(1).generate_table(get_correlated_tables()[0])
    t = np.array([r["temperature"] for r in rows])
    h = np.array([r["humidity"] for r in rows])
    p = np.array([r["pressure"] for r in rows])
    assert np.corrcoef(t, h)[0, 1] > 0.9
    assert np.corrcoef(t, p)[0, 1] < -0.8


def test_relsignature_union():
    a = RelSignature(frozenset({"a"}), frozenset({"p"}))
    b = RelSignature(frozenset({"b"}), frozenset())
    assert a.union(b, ["j"]).key == "a|b#j AND p"
