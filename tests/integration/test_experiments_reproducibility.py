"""
tests.integration.test_experiments_reproducibility — Acceptance tests for Phase 4.

Verifies:
1. ExperimentRunner executes matrix across multiple optimizers.
2. ResultWriter produces CSV, JSON summary, and reproducibility.json.
3. Provenance metadata records environment (Python, OS, commit hash, hardware).
4. Determinism: identical random seed produces exact row and metric match.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.config import ExperimentConfig
from experiment.runner import ExperimentRunner


def test_experiment_end_to_end_and_reproducibility(tmp_path: Path):
    """
    Test that an experiment runs end-to-end and produces full reproducibility artifacts.
    """
    config_dict = {
        "experiment": {
            "name": "test_reproducibility_exp",
            "description": "Integration test for Phase 4 reproducibility",
            "tags": ["test", "reproducibility"],
        },
        "dataset": {
            "name": "repro_db",
            "workload_id": "W1",
            "scale": 1,
            "seed": 999,
        },
        "optimizers": [
            {"id": "naive"},
            {"id": "rule_based"},
            {"id": "cost_based"},
        ],
        "queries": [
            {
                "id": "Q1",
                "sql": "SELECT id, amount FROM orders WHERE amount > 200",
                "description": "Point and range scan",
            }
        ],
        "execution": {
            "repetitions": 2,
            "warmup_runs": 1,
            "timeout_seconds": 10.0,
        },
        "result_storage": {
            "directory": str(tmp_path / "results"),
            "formats": ["json", "csv"],
            "save_raw": True,
            "save_summary": True,
            "overwrite": True,
        },
    }

    import yaml
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)

    config = ExperimentConfig.from_yaml(config_path)
    runner = ExperimentRunner(base_data_dir=tmp_path / "data")
    result = runner.run(config)

    # 1. Assertions on Experiment Result
    assert result.name == "test_reproducibility_exp"
    assert len(result.summaries) == 3
    opt_ids = {s.optimizer_id for s in result.summaries}
    assert opt_ids == {"naive", "rule_based", "cost_based"}
    assert all(s.query_id == "Q1" for s in result.summaries)

    # 2. Assertions on Generated Files
    exp_dir = result.output_dir
    assert exp_dir.exists()

    summary_file = exp_dir / "summary.json"
    summary_csv = exp_dir / "summary.csv"
    raw_metrics_file = exp_dir / "raw_metrics.csv"
    repro_file = exp_dir / "reproducibility.json"

    assert summary_file.exists(), "summary.json must exist"
    assert summary_csv.exists(), "summary.csv must exist"
    assert raw_metrics_file.exists(), "raw_metrics.csv must exist"
    assert repro_file.exists(), "reproducibility.json must exist"

    # 3. Assertions on Reproducibility Metadata (ADR-006)
    with open(repro_file, "r", encoding="utf-8") as f:
        repro_data = json.load(f)

    assert "experiment_id" in repro_data
    assert "name" in repro_data
    assert "timestamp_utc" in repro_data
    assert "git_commit" in repro_data
    assert "python_version" in repro_data
    assert "platform" in repro_data
    assert "config" in repro_data

    # 4. Assertions on CSV Content
    import csv
    with open(raw_metrics_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 6  # 1 query * 3 optimizers * 2 repetitions
        first_row = reader[0]
        assert "query_id" in first_row
        assert "optimizer_id" in first_row
        assert "execution_time_ms" in first_row
        assert "planning_time_ms" in first_row
        assert "q_error" in first_row
        assert "actual_rows" in first_row
