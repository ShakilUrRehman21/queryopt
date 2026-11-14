"""
apps.api.routes.experiment — Experiment execution and result inspection endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException

from apps.api.models import ExperimentRunApiRequest
from experiment.config import ExperimentConfig
from experiment.runner import ExperimentRunner

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("")
def list_experiments() -> list[dict[str, Any]]:
    """Lists all completed experiment result runs."""
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        return []

    experiments = []
    for item in results_dir.iterdir():
        if item.is_dir():
            repro_path = item / "reproducibility.json"
            summary_path = item / "summary.json"
            meta = {
                "name": item.name,
                "path": str(item),
                "has_summary": summary_path.exists(),
                "has_reproducibility": repro_path.exists(),
            }
            if repro_path.exists():
                try:
                    repro_data = json.loads(repro_path.read_text(encoding="utf-8"))
                    meta["timestamp_utc"] = repro_data.get("timestamp_utc")
                    meta["git_commit"] = repro_data.get("git_commit")
                    meta["description"] = repro_data.get("config", {}).get("experiment", {}).get("description")
                except Exception:
                    pass
            experiments.append(meta)

    return sorted(experiments, key=lambda x: x.get("timestamp_utc") or "", reverse=True)


@router.get("/{name}")
def get_experiment_details(name: str) -> dict[str, Any]:
    """Returns detailed metrics, summaries, and provenance metadata for an experiment."""
    exp_dir = Path("experiments/results") / name
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found.")

    details: dict[str, Any] = {"name": name}

    summary_file = exp_dir / "summary.json"
    if summary_file.exists():
        details["summary"] = json.loads(summary_file.read_text(encoding="utf-8"))

    repro_file = exp_dir / "reproducibility.json"
    if repro_file.exists():
        details["reproducibility"] = json.loads(repro_file.read_text(encoding="utf-8"))

    raw_metrics_file = exp_dir / "raw_metrics.json"
    if raw_metrics_file.exists():
        details["raw_metrics"] = json.loads(raw_metrics_file.read_text(encoding="utf-8"))

    return details


@router.post("/run")
def run_experiment(req: ExperimentRunApiRequest) -> dict[str, Any]:
    """Executes an experiment using a config name (e.g. 'smoke_test') or inline YAML."""
    if req.config_name:
        p = Path("experiments/configs") / f"{req.config_name}.yaml"
        if not p.exists():
            p = Path("experiments/configs") / req.config_name
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found for: {req.config_name}")
        config = ExperimentConfig.from_yaml(p)
    elif req.config_yaml:
        import yaml
        data = yaml.safe_load(req.config_yaml)
        config = ExperimentConfig.from_dict(data)
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'config_name' or 'config_yaml'")

    runner = ExperimentRunner()
    result = runner.run(config)

    summary_path = result.output_dir / "summary.json"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []

    return {
        "status": "SUCCESS",
        "experiment_name": result.name,
        "output_dir": str(result.output_dir),
        "summaries": summary_data,
    }
