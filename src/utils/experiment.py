"""Lightweight experiment bookkeeping.

Every run gets its own folder under the output directory containing:
    config.json   - the exact hyperparameters used
    metrics.csv   - per-epoch training history
    metrics.json  - final test metrics
    *.pt          - model checkpoints
    *.png         - figures

No experiment-tracking framework is used on purpose: plain JSON and a pandas
DataFrame are enough for a research project and stay easy to read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def create_run_dir(output_dir: Path, run_name: str) -> Path:
    """Create (and return) a fresh ``output_dir/run_name`` directory."""
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, config: dict) -> None:
    """Write the run configuration to ``config.json``."""
    path = Path(run_dir) / "config.json"
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def save_metrics(run_dir: Path, metrics: dict, file_name: str = "metrics.json") -> None:
    """Write a flat metrics dictionary to JSON."""
    path = Path(run_dir) / file_name
    path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")


def save_history(run_dir: Path, history: pd.DataFrame) -> Path:
    """Write the per-epoch training history to ``metrics.csv``."""
    path = Path(run_dir) / "metrics.csv"
    history.to_csv(path, index=False)
    return path
