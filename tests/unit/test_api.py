"""
tests.unit.test_api — Unit tests for QueryOpt Lab FastAPI endpoints.
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from apps.api.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "queryopt-lab-api"
    assert "version" in data


def test_schema_endpoint(client: TestClient):
    response = client.get("/api/schema")
    assert response.status_code == 200
    data = response.json()
    assert data["db_name"] == "default"
    assert isinstance(data["tables"], list)


def test_query_run(client: TestClient):
    from queryopt.engine import QueryEngine
    engine = QueryEngine(db_name="default")
    if "test_api_t" in engine.db.catalog.list_tables():
        engine.db.drop_table("test_api_t")

    # Create a small table and run a query
    client.post("/api/query/run", json={"sql": "CREATE TABLE test_api_t (id INTEGER, val TEXT)"})
    client.post("/api/query/run", json={"sql": "INSERT INTO test_api_t VALUES (1, 'hello')"})

    res = client.post("/api/query/run", json={
        "sql": "SELECT id, val FROM test_api_t WHERE id = 1",
        "optimizer_id": "cost_based",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert len(data["rows"]) == 1
    assert data["rows"][0] == [1, "hello"]


def test_query_explain(client: TestClient):
    res = client.post("/api/query/explain", json={
        "sql": "SELECT id, val FROM test_api_t WHERE id = 1",
        "optimizer_id": "cost_based",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "logical_plan_ascii" in data
    assert "physical_plan_ascii" in data
    assert "logical_graph" in data
    assert "physical_graph" in data
    assert len(data["logical_graph"]["nodes"]) > 0


def test_optimizer_compare(client: TestClient):
    res = client.post("/api/optimizer/compare", json={
        "sql": "SELECT id, val FROM test_api_t WHERE id = 1",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["results_match"] is True
    assert len(data["evaluations"]) == 3
    opt_ids = [e["optimizer_id"] for e in data["evaluations"]]
    assert "naive" in opt_ids
    assert "rule_based" in opt_ids
    assert "cost_based" in opt_ids


def test_benchmark_workloads_endpoint(client: TestClient):
    res = client.get("/api/benchmark/workloads")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 9
    workload_ids = [w["workload_id"] for w in data]
    assert "W1" in workload_ids
    assert "W7" in workload_ids


def test_experiments_endpoint(client: TestClient):
    res = client.get("/api/experiments")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)


def test_query_optimize_rewrite_endpoint(client: TestClient):
    # Test comma-join and wildcard detection
    res = client.post("/api/query/optimize-rewrite", json={
        "sql": "SELECT * FROM customers c, orders o WHERE c.id = o.customer_id AND o.total > 100",
        "dialect": "postgres"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["has_improvements"] is True
    assert "INNER JOIN orders" in data["optimized_sql"]
    assert len(data["anti_patterns"]) >= 1
    assert len(data["optimizations"]) >= 1
    assert len(data["index_recommendations"]) >= 1
    assert data["score_after"] > data["score_before"]

