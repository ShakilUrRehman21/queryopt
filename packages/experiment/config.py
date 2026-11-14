"""
experiment.config — Experiment configuration parsing and schema validation.

Loads experiment definitions from YAML files or Python dictionaries (ADR-006).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from common.errors import QueryOptError


class ExperimentConfigError(QueryOptError):
    """Raised when an experiment YAML configuration fails validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, error_code="CONFIG_ERROR")


@dataclass
class ExperimentMetadata:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    name: str = "default_benchmark"
    workload_id: str | None = None  # e.g. "W1", "W3"
    scale: int = 1
    seed: int = 42


@dataclass
class ExecutionConfig:
    repetitions: int = 5
    warmup_runs: int = 1
    timeout_seconds: float = 30.0
    # Cardinality feedback scope for the adaptive optimizer (RQ4):
    #   isolated — store is cleared before every (query, optimizer) cell, so the
    #              adaptive optimizer learns only from its own repetitions and
    #              repetition 1 is a cold start (clean learning curves);
    #   shared   — feedback accumulates across the whole experiment.
    feedback: str = "isolated"


@dataclass
class ResultStorageConfig:
    directory: str = "experiments/results/"
    formats: list[str] = field(default_factory=lambda: ["json", "csv"])
    save_raw: bool = True
    save_summary: bool = True
    overwrite: bool = False


@dataclass
class QueryConfig:
    id: str
    sql: str
    description: str = ""


@dataclass
class ExperimentConfig:
    """Full parsed configuration for an experiment run."""
    experiment: ExperimentMetadata
    dataset: DatasetConfig
    optimizers: list[str]
    queries: list[QueryConfig]
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    result_storage: ResultStorageConfig = field(default_factory=ResultStorageConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        if "experiment" not in data:
            raise ExperimentConfigError("Missing required 'experiment' section in configuration.")
        exp_data = data["experiment"]
        if "name" not in exp_data:
            raise ExperimentConfigError("Missing required 'experiment.name' field.")

        exp_meta = ExperimentMetadata(
            name=exp_data["name"],
            description=exp_data.get("description", ""),
            tags=list(exp_data.get("tags", [])),
        )

        ds_data = data.get("dataset", {})
        ds_cfg = DatasetConfig(
            name=ds_data.get("name", "benchmark_ds"),
            workload_id=ds_data.get("workload_id"),
            scale=int(ds_data.get("scale", 1)),
            seed=int(ds_data.get("seed", 42)),
        )

        optimizers = data.get("optimizers", ["rule_based", "cost_based"])
        if isinstance(optimizers, list) and optimizers and isinstance(optimizers[0], dict):
            # Format: [{'id': 'rule_based'}, {'id': 'cost_based'}]
            optimizers = [o.get("id", str(o)) for o in optimizers]

        # Parse queries
        raw_queries = data.get("queries", [])
        queries: list[QueryConfig] = []
        for q in raw_queries:
            if isinstance(q, dict):
                queries.append(
                    QueryConfig(
                        id=q.get("id", f"Q_{len(queries) + 1}"),
                        sql=q["sql"].strip(),
                        description=q.get("description", ""),
                    )
                )
            elif isinstance(q, str):
                queries.append(QueryConfig(id=f"Q_{len(queries) + 1}", sql=q.strip()))

        exec_data = data.get("execution", {})
        exec_cfg = ExecutionConfig(
            repetitions=int(exec_data.get("repetitions", 5)),
            warmup_runs=int(exec_data.get("warmup_runs", 1)),
            timeout_seconds=float(exec_data.get("timeout_seconds", 30.0)),
            feedback=str(exec_data.get("feedback", "isolated")),
        )
        if exec_cfg.feedback not in ("isolated", "shared"):
            raise ExperimentConfigError("execution.feedback must be 'isolated' or 'shared'.")

        res_data = data.get("result_storage", {})
        res_cfg = ResultStorageConfig(
            directory=res_data.get("directory", "experiments/results/"),
            formats=list(res_data.get("formats", ["json", "csv"])),
            save_raw=bool(res_data.get("save_raw", True)),
            save_summary=bool(res_data.get("save_summary", True)),
            overwrite=bool(res_data.get("overwrite", False)),
        )

        return cls(
            experiment=exp_meta,
            dataset=ds_cfg,
            optimizers=list(optimizers),
            queries=queries,
            execution=exec_cfg,
            result_storage=res_cfg,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        p = Path(path)
        if not p.exists():
            raise ExperimentConfigError(f"Experiment configuration file not found: {p}")
        try:
            content = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(content, dict):
                raise ExperimentConfigError(f"YAML file '{p}' must contain a key-value mapping.")
            return cls.from_dict(content)
        except yaml.YAMLError as e:
            raise ExperimentConfigError(f"Invalid YAML in '{p}': {e}")
