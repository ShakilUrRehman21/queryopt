"""Tests for the AI package using a scripted fake LLM (no network)."""
from __future__ import annotations

import json

import pytest

from ai.llm import LLMClient, LLMResponse, extract_json
from ai.llm_optimizer import LLMGuidedOptimizer
from ai.narrator import collect_facts, narrate
from ai.nl2sql import nl_to_sql
from ai.rewrite_verifier import verify_rewrite
from benchmark.generator import DatasetGenerator
from benchmark.workloads import create_workload
from queryopt.engine import QueryEngine

SQL3 = create_workload("W9").queries[0].sql


class FakeLLM(LLMClient):
    provider, model = "fake", "fake-1"

    def __init__(self, answers: list[str]) -> None:
        self.answers, self.prompts = list(answers), []

    def _call(self, system, prompt, max_tokens, temperature):
        self.prompts.append(prompt)
        return LLMResponse(text=self.answers.pop(0), provider=self.provider, model=self.model)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    eng = QueryEngine(data_dir=tmp_path_factory.mktemp("ai"), db_name="db")
    DatasetGenerator(seed=3).populate_database(eng.db, create_workload("W9").table_specs)
    return eng


@pytest.fixture(autouse=True)
def no_cache(monkeypatch):
    monkeypatch.setenv("QUERYOPT_LLM_CACHE", "false")


def _run(engine, opt, sql=SQL3):
    engine.optimizers["_t"] = opt
    r = engine.optimize_select(sql, "_t")
    return r, engine.run_optimized(r)


def test_extract_json_handles_fences():
    assert extract_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_llm_order_is_used_and_costed(engine):
    fake = FakeLLM([json.dumps({"order": ["zones", "sensors", "alerts"], "rationale": "small first"})])
    r, res = _run(engine, LLMGuidedOptimizer(client=fake))
    llm = r.metadata["llm"]
    assert llm["used"] and llm["order"] == ["zones", "sensors", "alerts"]
    assert llm["estimated_cost_ratio_vs_dp"] >= 0.999
    assert res.status.name == "SUCCESS"
    assert "RELATIONS" in fake.prompts[0]


def test_invalid_llm_order_falls_back_to_dp(engine):
    fake = FakeLLM([json.dumps({"order": ["zones", "nope"]})])
    r, res = _run(engine, LLMGuidedOptimizer(client=fake))
    assert not r.metadata["llm"]["used"]
    assert "permutation" in r.metadata["llm"]["fallback_reason"]
    assert res.status.name == "SUCCESS"


def test_offline_stand_in_is_labelled(engine):
    from ai.llm import OfflineClient

    r, _ = _run(engine, LLMGuidedOptimizer(client=OfflineClient()))
    assert r.metadata["llm"]["strategy"] == "greedy_offline"


def test_narrator_flags_correlated_misestimate(engine):
    sql = "SELECT id FROM sensors WHERE temperature > 85 AND humidity > 85"
    r = engine.optimize_select(sql, "cost_based")
    facts = collect_facts(sql, r, engine.run_optimized(r))
    n = narrate(facts)
    assert facts.worst and facts.worst.q_error > 3
    assert any("learned" in s for s in n["suggestions"])
    assert n["steps"][0].startswith("1. Read every row of **sensors**")


def test_nl2sql_repairs_with_engine_feedback(engine):
    fake = FakeLLM([
        json.dumps({"sql": "SELECT nope FROM sensors"}),
        json.dumps({"sql": "SELECT COUNT(id) FROM sensors WHERE temperature > 50", "explanation": "count"}),
    ])
    out = nl_to_sql(engine, "how many hot sensors?", client=fake)
    assert out["valid"] and len(out["attempts"]) == 2
    assert "ERROR" in fake.prompts[1]


def test_rewrite_verifier_detects_equivalence_and_difference(engine):
    base = "SELECT id FROM sensors WHERE temperature > 50"
    same = verify_rewrite(engine, base, "SELECT id FROM sensors WHERE 50 < temperature", repetitions=2)
    diff = verify_rewrite(engine, base, "SELECT id FROM sensors WHERE temperature > 60", repetitions=2)
    assert same["results_match"] and same["verdict"] in ("faster", "equivalent", "slower")
    assert diff["verdict"] == "not_equivalent"


def test_response_cache_replays(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERYOPT_LLM_CACHE", "true")
    monkeypatch.setenv("QUERYOPT_LLM_CACHE_PATH", str(tmp_path / "c.json"))
    fake = FakeLLM(['{"x": 1}'])
    assert not fake.complete("s", "p").cached
    assert fake.complete("s", "p").cached  # second call never reaches the fake


def test_ai_api_endpoints(tmp_path, monkeypatch):
    import dataclasses

    from starlette.testclient import TestClient

    import common.config as cfg
    from apps.api.main import app

    monkeypatch.setattr(cfg, "_default_config", dataclasses.replace(cfg.get_config(), data_dir=str(tmp_path)))
    with TestClient(app) as c:
        assert c.get("/api/ai/status").json()["optimizers"]
        arena = c.post("/api/ai/arena", json={"sql": SQL3}).json()
        assert arena["results_match"] and len(arena["entries"]) == 6
        ex = c.post("/api/ai/explain", json={"sql": SQL3, "use_llm": False}).json()
        assert ex["narrative"]["steps"]
        assert c.get("/api/ai/feedback").json()["entries"] > 0
        assert c.delete("/api/ai/feedback").status_code == 200
