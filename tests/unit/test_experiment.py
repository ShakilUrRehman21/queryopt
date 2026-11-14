"""
Unit tests for experiment framework (config, metrics, runner, storage).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.config import (
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ExperimentConfigError,
    ExperimentMetadata,
    QueryConfig,
)
from experiment.metrics import (
    MetricSummary,
    QueryRunMetric,
    QuerySummary,
    calculate_q_error,
)
from experiment.runner import ExperimentRunner
from experiment.storage import ResultWriter, build_reproducibility_record


class TestExperimentMetrics:

    def test_calculate_q_error_symmetric(self):
        """q-error is symmetric: q_error(100, 10) == q_error(10, 100) == 10.0."""
        q1 = calculate_q_error(100.0, 10.0)
        q2 = calculate_q_error(10.0, 100.0)
        assert q1 == pytest.approx(10.0)
        assert q2 == pytest.approx(10.0)

    def test_calculate_q_error_exact_match(self):
        """Exact estimate produces q-error of 1.0."""
        assert calculate_q_error(50.0, 50.0) == 1.0

    def test_metric_summary_compute(self):
        """MetricSummary computes accurate statistics."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        s = MetricSummary.compute(values)
        assert s.count == 5
        assert s.mean == 30.0
        assert s.median == 30.0
        assert s.min_val == 10.0
        assert s.max_val == 50.0
        assert s.stddev > 0.0

    def test_metric_summary_empty(self):
        s = MetricSummary.compute([])
        assert s.count == 0
        assert s.mean == 0.0


class TestExperimentConfig:

    def test_valid_config_from_dict(self):
        data = {
            "experiment": {"name": "test_exp", "description": "desc"},
            "dataset": {"name": "test_ds", "workload_id": "W1", "scale": 1},
            "optimizers": ["naive", "rule_based"],
            "queries": [{"id": "Q1", "sql": "SELECT id FROM orders"}],
            "execution": {"repetitions": 2, "warmup_runs": 1},
        }
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.experiment.name == "test_exp"
        assert cfg.dataset.workload_id == "W1"
        assert cfg.optimizers == ["naive", "rule_based"]
        assert len(cfg.queries) == 1
        assert cfg.execution.repetitions == 2

    def test_missing_experiment_raises(self):
        with pytest.raises(ExperimentConfigError, match="Missing required 'experiment'"):
            ExperimentConfig.from_dict({})

    def test_missing_experiment_name_raises(self):
        with pytest.raises(ExperimentConfigError, match="Missing required 'experiment.name'"):
            ExperimentConfig.from_dict({"experiment": {}})

    def test_from_yaml(self, tmp_path: Path):
        yaml_content = """
experiment:
  name: "yaml_exp"
optimizers:
  - naive
queries:
  - id: Q1
    sql: "SELECT 1"
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        cfg = ExperimentConfig.from_yaml(yaml_file)
        assert cfg.experiment.name == "yaml_exp"
        assert cfg.optimizers == ["naive"]


class TestExperimentStorage:

    def test_result_writer(self, tmp_path: Path):
        writer = ResultWriter(output_dir=tmp_path / "test_out")
        metrics = [
            QueryRunMetric(
                experiment_id="e1",
                query_id="q1",
                optimizer_id="naive",
                repetition=1,
                planning_time_ms=1.0,
                execution_time_ms=5.0,
                total_time_ms=6.0,
                estimated_cost=10.0,
                estimated_rows=20.0,
                actual_rows=20,
            )
        ]
        writer.write_raw_metrics(metrics, formats=["json", "csv"])

        assert (tmp_path / "test_out" / "raw_metrics.json").exists()
        assert (tmp_path / "test_out" / "raw_metrics.csv").exists()

        record = build_reproducibility_record("e1", "test_exp", {"opt": ["naive"]})
        writer.write_reproducibility(record)
        assert (tmp_path / "test_out" / "reproducibility.json").exists()


class TestExperimentRunner:

    def test_runner_execution(self, tmp_path: Path):
        """ExperimentRunner executes queries across optimizers and aggregates metrics."""
        cfg = ExperimentConfig(
            experiment=ExperimentMetadata(name="mini_runner_test"),
            dataset=DatasetConfig(workload_id="W1", scale=1, seed=42),
            optimizers=["naive", "cost_based"],
            queries=[
                QueryConfig(id="q1", sql="SELECT id, amount FROM orders WHERE amount > 500"),
            ],
            execution=ExecutionConfig(repetitions=2, warmup_runs=1),
        )

        runner = ExperimentRunner(base_data_dir=tmp_path / "dbs")
        result = runner.run(cfg)

        assert result.name == "mini_runner_test"
        # 1 query x 2 optimizers x 2 repetitions = 4 raw metrics
        assert len(result.raw_metrics) == 4
        # 1 query x 2 optimizers = 2 summaries
        assert len(result.summaries) == 2
        for s in result.summaries:
            assert s.execution_time_summary.count == 2
            assert s.actual_rows > 0
            assert s.q_error >= 1.0
