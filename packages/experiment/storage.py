"""
experiment.storage — Result persistence in JSON, CSV, and metadata formats.

Ensures every experiment run records full provenance and reproducibility metadata
(ADR-006).
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment.metrics import QueryRunMetric, QuerySummary


def get_git_commit_hash() -> str:
    """Extract current git commit hash if running in a repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def build_reproducibility_record(
    experiment_id: str,
    experiment_name: str,
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Capture environment, hardware, and configuration provenance."""
    return {
        "experiment_id": experiment_id,
        "name": experiment_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "queryopt_version": "0.2.0",
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "config": config_dict,
    }


class ResultWriter:
    """Persists experiment outputs to disk."""

    def __init__(self, output_dir: Path, overwrite: bool = False) -> None:
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_raw_metrics(self, metrics: list[QueryRunMetric], formats: list[str]) -> None:
        if "json" in formats:
            json_path = self.output_dir / "raw_metrics.json"
            data = [m.to_dict() for m in metrics]
            json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        if "csv" in formats and metrics:
            csv_path = self.output_dir / "raw_metrics.csv"
            keys = list(metrics[0].to_dict().keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for m in metrics:
                    writer.writerow(m.to_dict())

    def write_summary(self, summaries: list[QuerySummary], formats: list[str]) -> None:
        if "json" in formats:
            json_path = self.output_dir / "summary.json"
            data = [s.to_dict() for s in summaries]
            json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        if "csv" in formats and summaries:
            csv_path = self.output_dir / "summary.csv"
            keys = list(summaries[0].to_dict().keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for s in summaries:
                    writer.writerow(s.to_dict())

    def write_reproducibility(self, record: dict[str, Any]) -> None:
        path = self.output_dir / "reproducibility.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
